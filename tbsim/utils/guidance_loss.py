import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import tbsim.utils.geometry_utils as GeoUtils
from tbsim.utils.metrics import batch_detect_off_road
from tbsim.models.trace_helpers import (
    transform_agents_to_world,
)

### utils for choosing from samples ####

def choose_action_from_guidance(preds, obs_dict, guide_configs, guide_losses):
    B, N, T, _ = preds["positions"].size()
    # arbitrarily use the first sample as the action if no guidance given
    act_idx = torch.zeros((B), dtype=torch.long, device=preds["positions"].device)
    # choose sample closest to desired guidance
    accum_guide_loss = torch.stack([v for k,v in guide_losses.items()], dim=2)
    # each scene separately since may contain different guidance
    scount = 0
    for sidx in range(len(guide_configs)):
        scene_guide_cfg = guide_configs[sidx]
        ends = scount + len(scene_guide_cfg)
        scene_guide_loss = accum_guide_loss[..., scount:ends]
        scount = ends
        scene_mask = ~torch.isnan(torch.sum(scene_guide_loss, dim=[1,2]))
        scene_guide_loss = scene_guide_loss[scene_mask].cpu()
        scene_guide_loss = torch.nansum(scene_guide_loss, dim=-1)
        is_scene_level = np.array([guide_cfg.name in ['agent_collision', 'social_group'] for guide_cfg in scene_guide_cfg])
        if np.sum(is_scene_level) > 0: 
            # choose which sample minimizes at the scene level (where each sample is a "scene")
            scene_act_idx = torch.argmin(torch.sum(scene_guide_loss, dim=0))
        else:
            # each agent can choose the sample that minimizes guidance loss independently
            scene_act_idx = torch.argmin(scene_guide_loss, dim=-1)

        act_idx[scene_mask] = scene_act_idx.to(act_idx.device)

    return act_idx

def choose_action_from_gt(preds, obs_dict):
    B, N, T, _ = preds["positions"].size()
    # arbitrarily use the first sample as the action if no gt given
    act_idx = torch.zeros((B), dtype=torch.long, device=preds["positions"].device)
    if "target_positions" in obs_dict:
        print("DIFFUSER: WARNING using sample closest to GT from diffusion model!")
        # use the sample closest to GT
        # pred and gt may not be the same if gt is missing data at the end
        endT = min(T, obs_dict["target_positions"].size(1))
        pred_pos = preds["positions"][:,:,:endT]
        gt_pos = obs_dict["target_positions"][:,:endT].unsqueeze(1)
        gt_valid = obs_dict["target_availabilities"][:,:endT].unsqueeze(1).expand((B, N, endT))
        err = torch.norm(pred_pos - gt_pos, dim=-1)
        err[~gt_valid] = torch.nan # so doesn't affect
        ade = torch.nanmean(err, dim=-1) # B x N
        res_valid = torch.sum(torch.isnan(ade), dim=-1) == 0
        if torch.sum(res_valid) > 0:
            min_ade_idx = torch.argmin(ade, dim=-1)
            act_idx[res_valid] = min_ade_idx[res_valid]
    else:
        print('Could not choose sample based on GT, as no GT in data')

    return act_idx


############## GUIDANCE config ########################

class GuidanceConfig(object):
    def __init__(self, name, weight, params, agents, func=None):
        '''
        - name : name of the guidance function (i.e. the type of guidance), must be in GUIDANCE_FUNC_MAP
        - weight : alpha weight, how much affects denoising
        - params : guidance loss specific parameters
        - agents : agent indices within the scene to apply this guidance to. Applies to ALL if is None.
        - func : the function to call to evaluate this guidance loss value.
        '''
        assert name in GUIDANCE_FUNC_MAP, 'Guidance name must be one of: ' + ', '.join(map(str, GUIDANCE_FUNC_MAP.keys()))
        self.name = name
        self.weight = weight
        self.params = params
        self.agents = agents
        self.func = func

    @staticmethod
    def from_dict(config_dict):
        assert config_dict.keys() == {'name', 'weight', 'params', 'agents'}, \
                'Guidance config must include only [name, weight, params, agt_mask]. agt_mask may be None if applies to all agents in a scene'
        return GuidanceConfig(**config_dict)

    def __repr__(self):
        return '<\n%s\n>' % str('\n '.join('%s : %s' % (k, repr(v)) for (k, v) in self.__dict__.items()))

def verify_guidance_config_list(guidance_config_list):
    '''
    Returns true if there list contains some valid guidance that needs to be applied.
    Does not check to make sure each guidance dict is structured properly, only that
    the list structure is valid.
    '''
    assert len(guidance_config_list) > 0
    valid_guidance = False
    for guide in guidance_config_list:
        valid_guidance = valid_guidance or len(guide) > 0
    return valid_guidance

############## GUIDANCE functions ########################

class GuidanceLoss(nn.Module):
    '''
    Abstract guidance function. This is a loss (not a reward), i.e. guidance will seek to
    MINIMIZE the implemented function.
    '''
    def __init__(self):
        super().__init__()
        self.global_t = 0

    def init_for_batch(self, example_batch):
        '''
        Initializes this loss to be used repeatedly only for the given scenes/agents in the example_batch.
        e.g. this function could use the extents of agents or num agents in each scene to cache information
              that is used while evaluating the loss
        '''
        pass

    def update(self, global_t=None):
        '''
        Update any persistant state needed by guidance loss functions.
        - global_t : the current global timestep of rollout
        '''
        if global_t is not None:
            self.global_t = global_t

    def forward(self, x, data_batch, agt_mask=None):
        '''
        Computes and returns loss value.

        Inputs:
        - x : the current trajectory (B, N, T, 6) where N is the number of samples and 6 is (x, y, vel, yaw, acc, yawvel)
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        - agt_mask : size B boolean list specifying which agents to apply guidance to. Applies to ALL agents if is None.

        Output:
        - loss : (B, N) loss for each sample of each batch index. Final loss will be mean of this.
        '''
        raise NotImplementedError('Must implement guidance function evaluation')

class TargetSpeedLoss(GuidanceLoss):
    '''
    Agent should follow specific target speed.
    '''
    def __init__(self, target_speed, dt, mode='use_action'):
        super().__init__()
        self.target_speed = target_speed
        self.dt = dt
        self.mode = mode

    def forward(self, x, data_batch, agt_mask=None):
        if agt_mask is not None:
            x = x[agt_mask]
        cur_tgt_speed = self.target_speed
        if isinstance(cur_tgt_speed, list):
            assert x.size(0) == len(cur_tgt_speed)
            cur_tgt_speed = torch.tensor(cur_tgt_speed).to(x.device)
            
        if self.mode == 'use_action':
            # loss = torch.sigmoid(torch.mean(torch.abs(x[..., [2]] - target_speed)))
            # loss = torch.mean(torch.abs(x[..., [2]] - target_speed))
            loss = (x[..., 2] - cur_tgt_speed)**2
            loss = torch.mean(loss, dim=-1)
        elif self.mode == 'use_position':
            x_pos = x[..., :2]
            x_vel = (x_pos[:,:,1:] - x_pos[:,:,:-1]) / self.dt
            x_vel = torch.norm(x_vel, dim=-1)
            loss = (x_vel - cur_tgt_speed)**2
            loss = torch.mean(loss, dim=-1)
        
        return loss

class MinSpeedLoss(GuidanceLoss):
    def __init__(self, min_speed, dt):
        super().__init__()
        self.min_speed = min_speed
        self.dt = dt

    def forward(self, x, data_batch, agt_mask=None):
        if agt_mask is not None:
            x = x[agt_mask]
            
        x_pos = x[..., :2]
        x_vel = (x_pos[:,:,1:] - x_pos[:,:,:-1]) / self.dt
        x_vel = torch.norm(x_vel, dim=-1)
        loss = F.gelu(self.min_speed - x_vel) # only apply loss if below min_speed
        loss = torch.mean(loss**2, dim=-1)
        
        return loss

#
# Collision losses
#

class AgentCollisionLoss(GuidanceLoss):
    '''
    Agents should not collide with each other.
    NOTE: this assumes full control over the scene. 
    NOTE: this is not very efficient for num_scene_in_batch > 1
        since there will be two different agent collision losses, both of which
        compute the same thing just mask it differently. Really should apply 
        agent mask before computing anything, but this does not work if
        the agent_collision is only being applied to a subset of one scene.
    '''
    def __init__(self, num_disks=5, buffer_dist=0.2):
        '''
        - num_disks : the number of disks to use to approximate the agent for collision detection.
                        more disks improves accuracy
        - buffer_dist : additional space to leave between agents
        '''
        super().__init__()
        self.num_disks = num_disks
        self.buffer_dist = buffer_dist

        self.centroids = None
        self.penalty_dists = None
        self.scene_mask = None

    def init_for_batch(self, example_batch):
        '''
        Caches disks and masking ahead of time.
        '''
        # return 
        # pre-compute disks to approximate each agent
        data_extent = example_batch["extent"]
        self.centroids, agt_rad = self.init_disks(self.num_disks, data_extent) # B x num_disks x 2
        B = self.centroids.size(0)
        # minimum distance that two vehicle circle centers can be apart without collision (+ buffer)
        self.penalty_dists = agt_rad.view(B, 1).expand(B, B) + agt_rad.view(1, B).expand(B, B) + self.buffer_dist
        
        # pre-compute masking for vectorized pairwise distance computation
        self.scene_mask = self.init_mask(example_batch['scene_index'], self.centroids.device)

    def init_disks(self, num_disks, extents):
        NA = extents.size(0)
        agt_rad = extents[:, 1] / 2. # assumes lenght > width
        cent_min = -(extents[:, 0] / 2.) + agt_rad
        cent_max = (extents[:, 0] / 2.) - agt_rad
        # sample disk centroids along x axis
        cent_x = torch.stack([torch.linspace(cent_min[vidx].item(), cent_max[vidx].item(), num_disks) \
                                for vidx in range(NA)], dim=0).to(extents.device)
        centroids = torch.stack([cent_x, torch.zeros_like(cent_x)], dim=2)      
        return centroids, agt_rad

    def init_mask(self, batch_scene_index, device):
        _, data_scene_index = torch.unique_consecutive(batch_scene_index, return_inverse=True)
        scene_block_list = []
        scene_inds = torch.unique_consecutive(data_scene_index)
        for scene_idx in scene_inds:
            cur_scene_mask = data_scene_index == scene_idx
            num_agt_in_scene = torch.sum(cur_scene_mask)
            cur_scene_block = ~torch.eye(num_agt_in_scene, dtype=torch.bool)
            scene_block_list.append(cur_scene_block)
        scene_mask = torch.block_diag(*scene_block_list).to(device)
        return scene_mask

    def forward(self, x, data_batch, agt_mask=None):
        data_extent = data_batch["extent"]
        data_world_from_agent = data_batch["world_from_agent"]

        pos_pred = x[..., :2]
        yaw_pred = x[..., 3:4]

        pos_pred_global, yaw_pred_global = transform_agents_to_world(pos_pred, yaw_pred, data_world_from_agent)
        if agt_mask is not None:
            # only want gradient to backprop to agents being guided
            pos_pred_detach = pos_pred_global.detach().clone()
            yaw_pred_detach = yaw_pred_global.detach().clone()

            pos_pred_global = torch.where(agt_mask[:,None,None,None].expand_as(pos_pred_global),
                                          pos_pred_global,
                                          pos_pred_detach)
            yaw_pred_global = torch.where(agt_mask[:,None,None,None].expand_as(yaw_pred_global),
                                          yaw_pred_global,
                                          yaw_pred_detach)

        # create disks and transform to world frame (centroids)
        B, N, T, _ = pos_pred_global.size()
        if self.centroids is None or self.penalty_dists is None:
            centroids, agt_rad = self.init_disks(self.num_disks, data_extent) # B x num_disks x 2
            # minimum distance that two vehicle circle centers can be apart without collision (+ buffer)
            penalty_dists = agt_rad.view(B, 1).expand(B, B) + agt_rad.view(1, B).expand(B, B) + self.buffer_dist
        else:
            centroids, penalty_dists = self.centroids, self.penalty_dists
        centroids = centroids[:,None,None].expand(B, N, T, self.num_disks, 2)
        # to world
        s = torch.sin(yaw_pred_global).unsqueeze(-1)
        c = torch.cos(yaw_pred_global).unsqueeze(-1)
        rotM = torch.cat((torch.cat((c, s), dim=-1), torch.cat((-s, c), dim=-1)), dim=-2)
        centroids = torch.matmul(centroids, rotM) + pos_pred_global.unsqueeze(-2)

        # NOTE: assume each sample is a different scene for the sake of computing collisions
        if self.scene_mask is None:
            scene_mask = self.init_mask(data_batch['scene_index'], centroids.device)
        else:
            scene_mask = self.scene_mask

        # TODO technically we do not need all BxB comparisons
        #       only need the lower triangle of this matrix (no self collisions and only one way distance)
        #       but this may be slower to assemble than masking

        # TODO B could contain multiple scenes, could just pad each scene to the max_agents and compare MaxA x MaxA to avoid unneeded comparisons across scenes

        centroids = centroids.transpose(0,2) # T x NS x B x D x 2
        centroids = centroids.reshape((T*N, B, self.num_disks, 2))
        # distances between all pairs of circles between all pairs of agents
        cur_cent1 = centroids.view(T*N, B, 1, self.num_disks, 2).expand(T*N, B, B, self.num_disks, 2).reshape(T*N*B*B, self.num_disks, 2)
        cur_cent2 = centroids.view(T*N, 1, B, self.num_disks, 2).expand(T*N, B, B, self.num_disks, 2).reshape(T*N*B*B, self.num_disks, 2)
        pair_dists = torch.cdist(cur_cent1, cur_cent2).view(T*N*B*B, self.num_disks*self.num_disks)
        # get minimum distance over all circle pairs between each pair of agents
        pair_dists = torch.min(pair_dists, 1)[0].view(T*N, B, B)

        penalty_dists = penalty_dists.view(1, B, B)
        is_colliding_mask = torch.logical_and(pair_dists <= penalty_dists,
                                              scene_mask.view(1, B, B))

        # penalty is inverse normalized distance apart for those already colliding
        cur_penalties = 1.0 - (pair_dists / penalty_dists)
        # only compute loss where it's valid and colliding
        cur_penalties = torch.where(is_colliding_mask,
                                    cur_penalties,
                                    torch.zeros_like(cur_penalties))
                                        
        # summing over timesteps and all other agents to get B x N
        cur_penalties = cur_penalties.reshape((T, N, B, B))
        cur_penalties = cur_penalties.sum(0).sum(-1).transpose(0, 1)

        if agt_mask is not None:
            return cur_penalties[agt_mask]
        else:
            return cur_penalties


class MapCollisionLoss(GuidanceLoss):
    '''
    Agents should not go offroad.
    NOTE: this currently depends on the map that's also passed into the network.
            if the network map viewport is small and the future horizon is long enough,
            it may go outside the range of the map and then this is really inaccurate.
    '''
    def __init__(self, num_points_lw=(10, 10)):
        '''
        - num_points_lw : how many points will be sampled within each agent bounding box
                            to detect map collisions. e.g. (15, 10) will sample a 15 x 10 grid
                            of points where 15 is along the length and 10 along the width.
        '''
        super().__init__()
        self.num_points_lw = num_points_lw
        lwise = torch.linspace(-0.5, 0.5, self.num_points_lw[0])
        wwise = torch.linspace(-0.5, 0.5, self.num_points_lw[1])
        self.local_coords = torch.cartesian_prod(lwise, wwise)

    def gen_agt_coords(self, pos, yaw, lw, raster_from_agent):
        '''
        - pos : B x 2
        - yaw : B x 1
        - lw : B x 2
        '''
        B = pos.size(0)
        cur_loc_coords = self.local_coords.to(pos.device).unsqueeze(0).expand((B, -1, -1))
        # scale by the extents
        cur_loc_coords = cur_loc_coords * lw.unsqueeze(-2)

        # transform initial coords to given pos, yaw
        s = torch.sin(yaw).unsqueeze(-1)
        c = torch.cos(yaw).unsqueeze(-1)
        rotM = torch.cat((torch.cat((c, s), dim=-1), torch.cat((-s, c), dim=-1)), dim=-2)
        agt_coords_agent_frame = cur_loc_coords @ rotM + pos.unsqueeze(-2)
        
        # then transform to raster frame
        agt_coords_raster_frame = GeoUtils.transform_points_tensor(agt_coords_agent_frame, raster_from_agent)

        return agt_coords_agent_frame, agt_coords_raster_frame

    def forward(self, x, data_batch, agt_mask=None):   
        drivable_map = data_batch["drivable_map"]
        data_extent = data_batch["extent"]
        data_raster_from_agent = data_batch["raster_from_agent"]

        if agt_mask is not None:
            x = x[agt_mask]
            drivable_map = drivable_map[agt_mask]
            data_extent = data_extent[agt_mask]
            data_raster_from_agent = data_raster_from_agent[agt_mask]

        _, H, W = drivable_map.size()

        B, N, T, _ = x.size()
        traj = x.reshape((-1, 6)) # B*N*T x 6
        pos_pred = traj[:,:2]
        yaw_pred = traj[:, 3:4] 
        lw = data_extent[:,None,None].expand((B, N, T, 3)).reshape((-1, 3))[:,:2]
        diag_len = torch.sqrt(torch.sum(lw*lw, dim=-1))
        data_raster_from_agent = data_raster_from_agent[:,None,None].expand((B, N, T, 3, 3)).reshape((-1, 3, 3))

        # sample points within each agent to check if drivable
        agt_samp_pts, agt_samp_pix = self.gen_agt_coords(pos_pred, yaw_pred, lw, data_raster_from_agent)
        # agt_samp_pts = agt_samp_pts.reshape((B, N, T, -1, 2))
        agt_samp_pix = agt_samp_pix.reshape((B, N, T, -1, 2)).long().detach() # only used to query drivable map, not to compute loss
        # NOTE: this projects pixels outside the map onto the edge
        agt_samp_l = torch.clamp(agt_samp_pix[..., 0:1], 0, W-1)
        agt_samp_w = torch.clamp(agt_samp_pix[..., 1:2], 0, H-1)
        agt_samp_pix = torch.cat([agt_samp_l, agt_samp_w], dim=-1)

        # query these points in the drivable area to determine collision
        _, P, _ = agt_samp_pts.size()
        map_coll_mask = torch.isclose(batch_detect_off_road(agt_samp_pix, drivable_map), torch.ones((1)).to(agt_samp_pix.device))
        map_coll_mask = map_coll_mask.reshape((-1, P))

        # only apply loss to timesteps that are partially overlapping
        per_step_coll = torch.sum(map_coll_mask, dim=-1)
        overlap_mask = ~torch.logical_or(per_step_coll == 0, per_step_coll == P)

        overlap_coll_mask = map_coll_mask[overlap_mask]
        overlap_agt_samp = agt_samp_pts[overlap_mask]
        overlap_diag_len = diag_len[overlap_mask]

        #
        # The idea here: for each point that is offroad, we want to compute
        #   the minimum distance to a point that is on the road to give a nice
        #   gradient to push it back.
        #

        # compute dist mat between all pairs of points at each step
        # NOTE: the detach here is a very subtle but IMPORTANT point
        #       since these sample points are a function of the pos/yaw, if we compute
        #       the distance between them the gradients will always be 0, no matter how
        #       we change the pos and yaw the distance will never change. But if we detach
        #       one and compute distance to these arbitrary points we've selected, then
        #       we get a useful gradient.
        #           Moreover, it's also important the columns are the ones detached here!
        #       these correspond to the points that ARE colliding. So if we try to max
        #       distance b/w these and the points inside the agent, it will push the agent
        #       out of the offroad area. If we do it the other way it will pull the agent
        #       into the offroad (if we max the dist) or just be a small pull in the correct dir
        #       (if we min the dist).
        pt_samp_dist = torch.cdist(overlap_agt_samp, overlap_agt_samp.clone().detach())
        # get min dist just for points still on the road
        # so we mask out points off the road (this also removes diagonal for off-road points which excludes self distances)
        pt_samp_dist = torch.where(overlap_coll_mask.unsqueeze(-1).expand(-1, -1, P),
                                   torch.ones_like(pt_samp_dist)*np.inf,
                                   pt_samp_dist)
        pt_samp_min_dist_all = torch.amin(pt_samp_dist, dim=1) # previously masked rows, so compute min over cols
        # compute actual loss
        pt_samp_loss_all = 1.0 - (pt_samp_min_dist_all / overlap_diag_len.unsqueeze(1))
        # only want a loss for off-road points
        pt_samp_loss_offroad = torch.where(overlap_coll_mask,
                                               pt_samp_loss_all,
                                               torch.zeros_like(pt_samp_loss_all))

        overlap_coll_loss = torch.sum(pt_samp_loss_offroad, dim=-1)
        # expand back to all steps, other non-overlap steps will be zero
        all_coll_loss = torch.zeros((agt_samp_pts.size(0))).to(overlap_coll_loss.device)
        all_coll_loss[overlap_mask] = overlap_coll_loss
        all_coll_loss = all_coll_loss.reshape((B, N, T)).sum(-1)

        return all_coll_loss

class TargetPosAtTimeLoss(GuidanceLoss):
    '''
    Hit a specific waypoint at a specific time step (within the current planning horizon).
    '''
    def __init__(self, target_pos, target_time):
        '''
        - target_pos : (B,2) batch of positions to hit, B must equal the number of agents after applying mask in forward.
        - target_time: (B,) batch of times at which to hit the given positions
        '''
        super().__init__()
        self.set_target(target_pos, target_time)

    def set_target(self, target_pos, target_time):
        if isinstance(target_pos, torch.Tensor):
            self.target_pos = target_pos
        else:
            self.target_pos = torch.tensor(target_pos)
        if isinstance(target_time, torch.Tensor):
            self.target_time = target_time
        else:
            self.target_time = torch.tensor(target_time)

    def forward(self, x, data_batch, agt_mask=None):
        '''
        - x : the current trajectory (B, N, T, 6) where N is the number of samples and 6 is (x, y, vel, yaw, acc, yawvel)
        '''
        if agt_mask is not None:
            x = x[agt_mask]
        assert x.size(0) == self.target_pos.size(0)
        assert x.size(0) == self.target_time.size(0)
        
        x_pos = x[torch.arange(x.size(0)), :, self.target_time, :2]
        tgt_pos = self.target_pos.to(x_pos.device)[:,None] # (B,1,2)
        loss = torch.norm(x_pos - tgt_pos, dim=-1)

        return loss

class TargetPosLoss(GuidanceLoss):
    '''
    Hit a specific waypoint at some time step (within the current planning horizon).
    '''
    def __init__(self, target_pos, min_target_time=0.0):
        '''
        - target_pos : (B,2) batch of positions to hit, B must equal the number of agents after applying mask in forward.
        - min_target_time : float, only tries to hit the target after the initial min_target_time*horizon_num_steps of the trajectory
                            e.g. if = 0.5 then only the last half of the trajectory will attempt to go through target
        '''
        super().__init__()
        self.min_target_time = min_target_time
        self.set_target(target_pos)

    def set_target(self, target_pos):
        if isinstance(target_pos, torch.Tensor):
            self.target_pos = target_pos
        else:
            self.target_pos = torch.tensor(target_pos)

    def forward(self, x, data_batch, agt_mask=None):
        '''
        - x : the current trajectory (B, N, T, 6) where N is the number of samples and 6 is (x, y, vel, yaw, acc, yawvel)
        '''
        if agt_mask is not None:
            x = x[agt_mask]
        assert x.size(0) == self.target_pos.size(0)
        
        min_t = int(self.min_target_time*x.size(2))
        x_pos = x[:,:,min_t:,:2]
        tgt_pos = self.target_pos.to(x_pos.device)[:,None,None] # (B,1,1,2)
        dist = torch.norm(x_pos - tgt_pos, dim=-1)
        # give higher loss weight to the closest valid timesteps
        loss_weighting = F.softmin(dist, dim=-1)
        loss = loss_weighting * torch.sum((x_pos - tgt_pos)**2, dim=-1) # (B, N, T)
        loss = torch.mean(loss, dim=-1) # (B, N)

        return loss

def compute_progress_loss(pos_pred, tgt_pos, urgency,
                          tgt_time=None,
                          pref_speed=1.42,
                          dt=0.1,
                          min_progress_dist=0.5):
    '''
    Evaluate progress towards a goal that we want to hit at some point in the future.
    - pos_pred : (B x N x T x 2)
    - tgt_pos : (B x 2)
    - urgency : (B) in (0.0, 1.0]
    - tgt_time : [optional] (B) local target time, i.e. starting from the current t0 how many steps in the
                    future will we need to hit the target. If given, loss is computed to cover the distance
                    necessary to hit the goal at the given time
    - pref_speed: speed used to determine how much distance should be covered in a time interval
    - dt : step interval of the trajectories
    - min_progress_dist : float (in meters). if not using tgt_time, the minimum amount of progress that should be made in
                            each step no matter what the urgency is
    '''

    # distance from final trajectory timestep to the goal position
    final_dist = torch.norm(pos_pred[:,:,-1] - tgt_pos[:,None], dim=-1)

    if tgt_time is not None:
        #
        # have a target time: distance covered is based on arrival time
        #
        # distance of straight path from current pos to goal at the average speed
        goal_dist = tgt_time[:,None] * dt * pref_speed
        # factor in urgency (shortens goal_dist since we can't expect to always go on a straight path)
        goal_dist = goal_dist * (1.0 - urgency[:,None])
        # only apply loss if above the goal distance
        progress_loss = F.relu(final_dist - goal_dist)
    else:
        #
        # don't have a target time: distance covered based on making progress
        #       towards goal with the specified urgency
        #
        # following straight line path from current pos to goal
        max_horizon_dist = pos_pred.size(2) * dt * pref_speed
        # at max urgency, want to cover distance of this straight line path
        # at min urgency, just make minimum progress
        goal_dist = torch.maximum(urgency * max_horizon_dist, torch.tensor([min_progress_dist]).to(urgency.device))

        init_dist = torch.norm(pos_pred[:,:,0] - tgt_pos[:,None], dim=-1)
        progress_dist = init_dist - final_dist
        # only apply loss if less progress than goal
        progress_loss = F.relu(goal_dist[:,None] - progress_dist)

    return progress_loss

class GlobalTargetPosAtTimeLoss(GuidanceLoss):
    '''
    Hit a specific waypoint at a specific time step (in some future planning horizon).
    '''
    def __init__(self, target_pos, target_time, urgency, pref_speed=1.42, dt=0.1):
        '''
        - target_pos : (B,2) batch of GLOBAL positions to hit, B must equal the number of agents after applying mask in forward.
        - target_time: (B,) batch of GLOBAL times at which to hit the given positions
        - urgency: (B,) batch of [0.0, 1.0] urgency factors for each agent
                        The larger the urgency, the closer the agent will try to
                        to be at each planning step. This is used to scale the goal distance, i.e.
                        with urgency of 0.0, the agent will try to be close enough to the target
                        that they can take a straight path and get there on time. With urgency 1.0,
                        the agent will try to already be at the goal at the last step of every planning step.
        - pref_speed: float, speed used to determine how much distance should be covered in a time interval
                        by default 1.42 m/s (https://en.wikipedia.org/wiki/Preferred_walking_speed)
        - dt : of the timesteps that will be passed in (i.e. the diffuser model)
        '''
        super().__init__()
        self.set_target(target_pos, target_time)
        self.urgency = torch.tensor(urgency)
        self.pref_speed = pref_speed
        self.dt = dt
        # create local loss to use later when within reach
        #       will update target_pos/time later as necessary
        self.local_tgt_loss = TargetPosAtTimeLoss(target_pos, target_time)

    def set_target(self, target_pos, target_time):
        self.target_pos = torch.tensor(target_pos)
        self.target_time = torch.tensor(target_time)

    def forward(self, x, data_batch, agt_mask=None):
        '''
        - x : the current trajectory (B, N, T, 6) where N is the number of samples and 6 is (x, y, vel, yaw, acc, yawvel)
        '''
        agent_from_world = data_batch["agent_from_world"]
        if agt_mask is not None:
            x = x[agt_mask]
            agent_from_world = agent_from_world[agt_mask]
        assert x.size(0) == self.target_pos.size(0)
        assert x.size(0) == self.target_time.size(0)
        assert x.size(0) == self.urgency.size(0)

        # transform world targets to agent frame
        local_target_pos = GeoUtils.transform_points_tensor(self.target_pos[:,None].to(x.device), agent_from_world)[:,0]

        # decide which agents need progress loss vs. exact target loss
        local_target_time = self.target_time.to(x.device) - self.global_t
        horizon_len = x.size(2)
        # if within planning horizon but hasn't been passed yet
        exact_mask = torch.logical_and(local_target_time < horizon_len, local_target_time >= 0)
        # apply progress loss if not within planning horizon yet and hasn't been passed
        prog_mask = torch.logical_and(~exact_mask, local_target_time >= 0)

        loss = torch.zeros((x.size(0), x.size(1))).to(x)
        # progress loss
        num_exact = torch.sum(exact_mask)
        if num_exact != x.size(0):
            pos_pred = x[..., :2]
            progress_loss = compute_progress_loss(pos_pred[prog_mask],
                                                  local_target_pos[prog_mask],
                                                  self.urgency[prog_mask].to(x.device),
                                                  local_target_time[prog_mask],
                                                  self.pref_speed,
                                                  self.dt)
            loss[prog_mask] = progress_loss
        # exact target loss
        if  num_exact > 0:
            exact_local_tgt_pos = local_target_pos[exact_mask]
            exact_local_tgt_time = local_target_time[exact_mask]
            self.local_tgt_loss.set_target(exact_local_tgt_pos, exact_local_tgt_time)
            exact_loss = self.local_tgt_loss(x[exact_mask], None, None) # shouldn't need data_batch or agt_mask
            loss[exact_mask] = exact_loss

        return loss

class GlobalTargetPosLoss(GuidanceLoss):
    '''
    Hit a specific waypoint at some time in the future.
    '''
    def __init__(self, target_pos, urgency, pref_speed=1.42, dt=0.1, min_progress_dist=0.5):
        '''
        - target_pos : (B,2) batch of GLOBAL positions to hit, B must equal the number of agents after applying mask in forward.
        - urgency: (B,) batch of [0.0, 1.0] urgency factors for each agent
                        urgency in this case means how much of the maximal possible distance should
                        be covered in a single planning horizon. If urgency is 1.0 the agent
                        will shoot for a straight line path to the target. If it is 0.0 it will just
                        try to make the minimal amount of progress at each plan.
        - pref_speed: float, speed used to determine how much distance should be covered in a time interval
                        by default 1.42 m/s (https://en.wikipedia.org/wiki/Preferred_walking_speed)
        - dt : of the timesteps that will be passed in (i.e. the diffuser model)
        - min_progress_dist : minimum distance that should be covered in each plan no matter what the urgency is
        '''
        super().__init__()
        self.set_target(target_pos)
        self.urgency = torch.tensor(urgency)
        self.pref_speed = pref_speed
        self.dt = dt
        self.min_progress_dist = min_progress_dist
        # create local loss to use later when within reach
        #       will update target_pos/time later as necessary
        self.local_tgt_loss = TargetPosLoss(target_pos)

    def set_target(self, target_pos):
        self.target_pos = torch.tensor(target_pos)

    def forward(self, x, data_batch, agt_mask=None):
        '''
        - x : the current trajectory (B, N, T, 6) where N is the number of samples and 6 is (x, y, vel, yaw, acc, yawvel)
        '''
        agent_from_world = data_batch["agent_from_world"]
        if agt_mask is not None:
            x = x[agt_mask]
            agent_from_world = agent_from_world[agt_mask]
        assert x.size(0) == self.target_pos.size(0)
        assert x.size(0) == self.urgency.size(0)

        # transform world targets to agent frame
        local_target_pos = GeoUtils.transform_points_tensor(self.target_pos[:,None].to(x.device), agent_from_world)[:,0]

        # decide which agents need progress loss vs. exact target loss
        # if agent can progress along straight line at preferred speed
        #       and arrive at target within the horizon, consider it in range
        horizon_len = x.size(2)
        single_horizon_dist = horizon_len * self.dt * self.pref_speed
        # single_horizon_dist *= 0.35 # optionally stay on progress loss for longer (good for combining with PACER)
        local_target_dist = torch.norm(local_target_pos, dim=-1)
        exact_mask = local_target_dist < single_horizon_dist
        prog_mask = ~exact_mask

        loss = torch.zeros((x.size(0), x.size(1))).to(x)
        # progress loss
        num_exact = torch.sum(exact_mask)
        if num_exact != x.size(0):
            pos_pred = x[..., :2]
            progress_loss = compute_progress_loss(pos_pred[prog_mask],
                                                  local_target_pos[prog_mask],
                                                  self.urgency[prog_mask].to(x.device),
                                                  None,
                                                  self.pref_speed,
                                                  self.dt,
                                                  self.min_progress_dist)
            loss[prog_mask] = progress_loss
        # exact target loss
        if  num_exact > 0:
            exact_local_tgt_pos = local_target_pos[exact_mask]
            self.local_tgt_loss.set_target(exact_local_tgt_pos)
            exact_loss = self.local_tgt_loss(x[exact_mask], None, None) # shouldn't need data_batch or agt_mask
            loss[exact_mask] = exact_loss

        return loss

class SocialGroupLoss(GuidanceLoss):
    '''
    Agents should move together.
    NOTE: this assumes full control over the scene. 
    '''
    def __init__(self, leader_idx=0, social_dist=1.5, cohesion=0.8):
        '''
        - leader_idx : index to serve as the leader of the group (others will follow them). This is the index in the scene, not the index within the specific social group.
        - social_dist : float, meters, How close members of the group will stand to each other.
        - cohesion : float [0.0, 1.0], at 1.0 essentially all group members try to be equidistant
                                            at 0.0 try to maintain distance only to closest neighbor and could get detached from rest of group
        '''
        super().__init__()
        self.leader_idx = leader_idx
        self.social_dist = social_dist
        assert cohesion >= 0.0 and cohesion <= 1.0
        self.random_neighbor_p = cohesion

    def forward(self, x, data_batch, agt_mask=None):
        data_world_from_agent = data_batch["world_from_agent"]
        pos_pred = x[..., :2]
        yaw_pred = x[..., 3:4]
        agt_idx = torch.arange(pos_pred.shape[0]).to(pos_pred.device)
        if agt_mask is not None:
            data_world_from_agent = data_world_from_agent[agt_mask]
            pos_pred = pos_pred[agt_mask]
            yaw_pred = yaw_pred[agt_mask]
            agt_idx = agt_idx[agt_mask]

        pos_pred_global, _ = transform_agents_to_world(pos_pred, yaw_pred, data_world_from_agent)

        # NOTE here we detach the leader pos, so that their motion is not affected by trying to stay close to the group
        #       this is so the group makes progress by following rather than just trying to be close together
        leader_mask = (agt_idx == self.leader_idx)[:,None,None,None].expand_as(pos_pred_global)
        pos_pred_global = torch.where(leader_mask, pos_pred_global.detach(), pos_pred_global)

        # print(leader_pos.size())
        # print(other_pos.size())

        # compute closest distance to others in social group
        B, N, T, _ = pos_pred_global.size()
        flat_pos_pred = pos_pred_global.transpose(0, 2).reshape((T*N, B, 2))
        flat_dist = torch.cdist(flat_pos_pred, flat_pos_pred) # T*N x B x B
        self_mask = torch.eye(B, device=flat_dist.device).unsqueeze(0).expand_as(flat_dist)
        flat_dist = torch.where(self_mask.bool(), np.inf*self_mask, flat_dist)  # mask out self-distances
        # pairs with neighbors based purely on min distance
        min_neighbor = torch.argmin(flat_dist, dim=-1)

        # randomly switch some closest neighbors to make more cohesive (but not to self)
        #       the idea is to avoid degenerate case where subset of agents create a connected component
        #       in nearest neighbor graph and drift from the rest of the group.
        # creates 2D matrix with self indices missing
        #   i.e for 4 agents [1, 2, 3]
        #                    [0, 2, 3]
        #                    [0, 1, 3]
        #                    [0, 2, 2]
        neighbor_choices = torch.arange(B)[None].expand((B,B)).masked_select(~torch.eye(B, dtype=bool)).view(B, B - 1).to(min_neighbor.device)
        neighbor_choices = neighbor_choices.unsqueeze(0).expand((T*N, B, B-1))
        # randomly sample one with p = self.random_neighbor_p
        rand_neighbor = torch.gather(neighbor_choices, 2, torch.randint(0, B-1, (T*N, B, 1)).to(neighbor_choices.device))[:,:,0]
        drop_mask = torch.rand((T*N, B)).to(min_neighbor.device) < self.random_neighbor_p
        neighbor_idx = torch.where(drop_mask,
                                   rand_neighbor,
                                   min_neighbor)

        # want assigned neighbor dist to be the desired social distance
        neighbor_dist = torch.gather(flat_dist, 2, neighbor_idx.unsqueeze(-1))[..., 0]
        neighbor_dist = neighbor_dist.reshape((T, N, B)).transpose(0, 2) # B, N, T
        loss = torch.mean((neighbor_dist - self.social_dist)**2, dim=-1)
        
        return loss
        
class AmpValueLoss(GuidanceLoss):
    '''
    Maximize value function (future rewards) from AMP.
    '''
    def __init__(self, value_func):
        '''
        - value_func : handle to query value function which takes in trajectories
        '''
        super().__init__()
        self.value_func = value_func

    def forward(self, x, data_batch, agt_mask=None):
        '''
        Output:
        - loss : (B, N) loss for each sample of each batch index. Final loss will be mean of this.
        '''
        data_world_from_agent = data_batch["world_from_agent"]
        pos_pred = x[..., :2]
        yaw_pred = x[..., 3:4]

        cur_pos_global = data_batch["centroid"]

        pos_pred_global, yaw_pred_global = transform_agents_to_world(pos_pred, yaw_pred, data_world_from_agent)
        if agt_mask is not None:
            pos_pred_global = pos_pred_global[agt_mask]
            yaw_pred_global = yaw_pred_global[agt_mask]
            cur_pos_global = cur_pos_global[agt_mask]

        B, N, _, _ = pos_pred_global.size()
        pos_traj = torch.cat([cur_pos_global[:,None,None].expand((B, N, 1, 2)), pos_pred_global], dim=2)

        val = self.value_func(pos_traj)
        loss = torch.exp(-val) # reward to loss
        return loss

############## GUIDANCE utilities ########################

GUIDANCE_FUNC_MAP = {
    'target_speed' : TargetSpeedLoss,
    'agent_collision' : AgentCollisionLoss,
    'map_collision' : MapCollisionLoss,
    'target_pos_at_time' : TargetPosAtTimeLoss,
    'target_pos' : TargetPosLoss,
    'global_target_pos_at_time' : GlobalTargetPosAtTimeLoss,
    'global_target_pos' : GlobalTargetPosLoss,
    'social_group' : SocialGroupLoss,
    'min_speed' : MinSpeedLoss,
    'amp_value' : AmpValueLoss
}

class DiffuserGuidance(object):
    '''
    Handles initializing guidance functions and computing gradients at test-time.
    '''
    def __init__(self, guidance_config_list, example_batch=None):
        '''
        - example_batch [optional] - if this guidance will only be used on a single batch repeatedly,
                                    i.e. the same set of scenes/agents, an example data batch can
                                    be passed in a used to init some guidance making test-time more efficient.
        '''
        self.num_scenes = len(guidance_config_list)
        assert self.num_scenes > 0, "Guidance config list must include list of guidance for each scene"
        self.guide_configs = [[]]*self.num_scenes
        for si in range(self.num_scenes):
            if len(guidance_config_list[si]) > 0:
                self.guide_configs[si] = [GuidanceConfig.from_dict(cur_cfg) for cur_cfg in guidance_config_list[si]]
                # initialize each guidance function
                for guide_cfg in self.guide_configs[si]:
                    guide_cfg.func = GUIDANCE_FUNC_MAP[guide_cfg.name](**guide_cfg.params)
                    if example_batch is not None:
                        guide_cfg.func.init_for_batch(example_batch)


    def compute_guidance_loss(self, x_loss, data_batch):
        '''
        Evaluates all guidance losses and total and individual values.
        - x_loss: (B, N, T, 6) the trajectory to use to compute losses and 6 is (x, y, vel, yaw, acc, yawvel)
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        '''
        bsize, num_samp, _, _ = x_loss.size()
        guide_losses = dict()
        loss_tot = 0.0
        # NOTE: unique_consecutive is important here to avoid sorting by torch.unique which may shuffle the scene ordering
        #       and breaks correspondence with guide_configs
        _, local_scene_index = torch.unique_consecutive(data_batch['scene_index'], return_inverse=True)
        for si in range(self.num_scenes):
            cur_guide = self.guide_configs[si]
            if len(cur_guide) > 0:
                # mask out non-current current scene
                for gidx, guide_cfg in enumerate(cur_guide):
                    agt_mask = local_scene_index == si
                    if guide_cfg.agents is not None:
                        # mask out non-requested agents within the scene
                        cur_scene_inds = torch.nonzero(agt_mask, as_tuple=True)[0]
                        agt_mask_inds = cur_scene_inds[guide_cfg.agents]
                        agt_mask = torch.zeros_like(agt_mask)
                        agt_mask[agt_mask_inds] = True
                    # compute loss
                    cur_loss = guide_cfg.func(x_loss, data_batch,
                                            agt_mask=agt_mask)
                    indiv_loss = torch.ones((bsize, num_samp)).to(cur_loss.device) * np.nan # return indiv loss for whole batch, not just masked ones
                    indiv_loss[agt_mask] = cur_loss.detach().clone()
                    guide_losses[guide_cfg.name + '_scene_%03d_%02d' % (si, gidx)] = indiv_loss
                    loss_tot = loss_tot + torch.mean(cur_loss) * guide_cfg.weight

        return loss_tot, guide_losses

    def update(self, **kwargs):
        for si in range(self.num_scenes):
            cur_guide = self.guide_configs[si]
            if len(cur_guide) > 0:
                for guide_cfg in cur_guide:
                    guide_cfg.func.update(**kwargs)