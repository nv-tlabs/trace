
from collections import defaultdict

import torch
import numpy as np

from tbsim.utils.geometry_utils import transform_points_tensor, transform_yaw, detect_collision, batch_nd_transform_points_np
from tbsim.envs.env_metrics import EnvMetrics
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.batch_utils import batch_utils
import tbsim.utils.metrics as Metrics

from tbsim.utils.trajdata_utils import get_raster_pix2m

############## GUIDANCE METRICS ########################

#
# NOTE: each metric should handle ONLY THE SPECIFIED SCENE and AGENTS
#       specified by the guidance configuration
#

class GuidanceMetric(EnvMetrics):
    def __init__(self, scene_idx, params, agents, num_scenes):
        '''
        - scene_idx : which scene index this guidance is used for (within each batch, i.e. the LOCAL scene index)
        - params : param dict that defines the guidance behavior
        - agents : which agents in the scene the guidance was used on.
        - num_scenes : number of total scenes in a batch (i.e. how many will be in state_info when passed in)
        '''
        self._df = None
        self._scene_ts = defaultdict(lambda:0)

        self.scene_idx = scene_idx
        self.params = params
        self.agents = agents
        self.num_scenes = num_scenes

        self.reset()

class TargetSpeedGuidance(GuidanceMetric):
    """Compute the average speed satisfaction of the agents"""
    def reset(self):
        self.target_speed = self.params['target_speed']
        self._per_step = []

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray):
        """Compute per-agent and per-scene average speed"""
        speed = np.array(state_info['curr_speed'])
        # filter for specified scene and agents only
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_index"]), return_inverse=True)[1].numpy()
        scene_mask = local_scene_index == self.scene_idx
        speed = speed[scene_mask]
        if self.agents is not None:
            speed = speed[self.agents]
        speed_deviation = np.abs(speed - self.target_speed)
        return speed_deviation

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        met = self.compute_per_step(state_info, all_scene_index)
        met = np.mean(met)
        # must return metric for all scenes
        all_scene_met = np.ones((len(all_scene_index)))*np.nan
        all_scene_met[self.scene_idx] = met
        self._per_step.append(all_scene_met)

    def get_episode_metrics(self):
        met = np.stack(self._per_step, axis=0).transpose((1, 0))  # [num_scenes, num_steps]
        met = np.mean(met, axis=1)
        return met

class AgentCollisionGuidance(GuidanceMetric):
    '''
    Similar to regular collision rate, but only operates on a single scene
    and returns the rate for the specified agents in the guidance config.
    '''
    def reset(self):
        self._per_step = []

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray):
        """Compute per-agent collision rate and type"""
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_index"]), return_inverse=True)[1].numpy()
        scene_mask = local_scene_index == self.scene_idx
        pos_cur_scene = state_info["centroid"][scene_mask]
        yaw_cur_scene = state_info["yaw"][scene_mask]
        extent_cur_scene = state_info["extent"][..., :2][scene_mask]

        num_agents = pos_cur_scene.shape[0]

        coll_rates = np.zeros(num_agents)

        # compute collision rate
        for j in range(num_agents):
            other_agent_mask = np.arange(num_agents) != j
            coll = detect_collision(
                ego_pos=pos_cur_scene[j],
                ego_yaw=yaw_cur_scene[j],
                ego_extent=extent_cur_scene[j],
                other_pos=pos_cur_scene[other_agent_mask],
                other_yaw=yaw_cur_scene[other_agent_mask],
                other_extent=extent_cur_scene[other_agent_mask]
            )
            if coll is not None:
                coll_rates[j] = 1.

        # compute per-scene collision counts (for visualization purposes)
        if self.agents is not None:
            # mask to only the relevant agents
            coll_rates = coll_rates[self.agents]
        coll_counts = np.sum(coll_rates)

        return coll_rates, coll_counts

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        met_all, _ = self.compute_per_step(state_info, all_scene_index)
        self._per_step.append(met_all)

    def get_episode_metrics(self):
        met_all_steps = np.stack(self._per_step, axis=1)
        met_guide_scene = np.mean(np.amax(met_all_steps, axis=1))
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met_guide_scene
        return all_scene_met

class AgentCollisionGuidanceDisk(GuidanceMetric):
    '''
    Similar to regular metric, but only counts collision if a single disk that approximates 
    each agent overlaps (including the BUFFER DIST if desired).
    '''
    def __init__(self, scene_idx, params, agents, num_scenes, use_buffer_dist=False):
        super().__init__(scene_idx, params, agents, num_scenes)
        self.use_buffer_dist = use_buffer_dist

    def reset(self):
        self._per_step = []

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray):
        """Compute per-agent collision rate and type"""
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_index"]), return_inverse=True)[1].numpy()
        scene_mask = local_scene_index == self.scene_idx
        pos_cur_scene = state_info["centroid"][scene_mask]
        extent_cur_scene = state_info["extent"][..., :2][scene_mask]
        rad_cur_scene = np.amin(extent_cur_scene, axis=-1) / 2.0

        num_agents = pos_cur_scene.shape[0]

        coll_rates = np.zeros(num_agents)

        # compute collision rate
        for j in range(num_agents):
            # computes if j collides with ANY other agent. only counts single collision
            #       to be consistent with other coll metric.
            other_agent_mask = np.arange(num_agents) != j
            neighbor_dist = np.linalg.norm(pos_cur_scene[j:j+1] - pos_cur_scene[other_agent_mask], axis=-1)
            min_allowed_dist = rad_cur_scene[j] + rad_cur_scene[other_agent_mask]
            if self.use_buffer_dist:
                min_allowed_dist = min_allowed_dist + self.params['buffer_dist']
            coll = np.sum(neighbor_dist < min_allowed_dist) > 0
            if coll:
                coll_rates[j] = 1.

        # compute per-scene collision counts (for visualization purposes)
        if self.agents is not None:
            # mask to only the relevant agents
            coll_rates = coll_rates[self.agents]
        coll_counts = np.sum(coll_rates)

        return coll_rates, coll_counts

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        met_all, _ = self.compute_per_step(state_info, all_scene_index)
        self._per_step.append(met_all)

    def get_episode_metrics(self):
        met_all_steps = np.stack(self._per_step, axis=1)
        met_guide_scene = np.mean(np.amax(met_all_steps, axis=1))
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met_guide_scene
        return all_scene_met

class SocialDistanceGuidance(AgentCollisionGuidanceDisk):
    def __init__(self, scene_idx, params, agents, num_scenes):
        super().__init__(scene_idx, params, agents, num_scenes, use_buffer_dist=True)

class TargetPosGuidance(GuidanceMetric):
    """
    How well target pos (at any timestep) are met.
    """
    def reset(self):
        self.locs = np.array(self.params['target_pos'])
        self._agent_from_world = None
        self._cur_min_dist = np.ones((self.locs.shape[0]))*np.inf

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray):
        ''' stores all given positions for needed agents '''
        # which agent indices belong to each scene (list of np arrays with agent inds)
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_index"]), return_inverse=True)[1].numpy()
        scene_mask = local_scene_index == self.scene_idx
        pos_cur_scene = state_info["centroid"][scene_mask]
        if self._agent_from_world is None:
            # only want to record this at the first step (the planning step)
            #       b/c this is what the target is in.
            #       TODO: when the target is global, need to change this
            self._agent_from_world = state_info["agent_from_world"][scene_mask]
        pos_cur_scene = batch_nd_transform_points_np(pos_cur_scene[:,None], self._agent_from_world)[:,0]
        if self.agents is not None:
            pos_cur_scene = pos_cur_scene[self.agents]
        cur_dist = np.linalg.norm(pos_cur_scene - self.locs, axis=-1)
        new_min_mask = cur_dist < self._cur_min_dist
        self._cur_min_dist[new_min_mask] = cur_dist[new_min_mask]

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        self.compute_per_step(state_info, all_scene_index)

    def get_episode_metrics(self):
        met = np.mean(self._cur_min_dist)
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met
        return all_scene_met

class GlobalTargetPosGuidance(GuidanceMetric):
    """
    How well global target pos (at any timestep) are met.
    """
    def reset(self):
        self.locs = np.array(self.params['target_pos'])
        self._cur_min_dist = np.ones((self.locs.shape[0]))*np.inf

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray):
        ''' stores all given positions for needed agents '''
        # which agent indices belong to each scene (list of np arrays with agent inds)
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_index"]), return_inverse=True)[1].numpy()
        scene_mask = local_scene_index == self.scene_idx
        pos_cur_scene = state_info["centroid"][scene_mask]
        if self.agents is not None:
            pos_cur_scene = pos_cur_scene[self.agents]
        cur_dist = np.linalg.norm(pos_cur_scene - self.locs, axis=-1)
        new_min_mask = cur_dist < self._cur_min_dist
        self._cur_min_dist[new_min_mask] = cur_dist[new_min_mask]

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        self.compute_per_step(state_info, all_scene_index)

    def get_episode_metrics(self):
        met = np.mean(self._cur_min_dist)
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met
        return all_scene_met

class ConstraintGuidance(GuidanceMetric):
    """
    How well constraint (waypoints at specific time) are met.
    """
    def reset(self):
        if 'locs' in self.params and 'times' in self.params:
            # true hard constraint
            self.locs = np.array(self.params['locs'])
            self.times = np.array(self.params['times'])
        elif 'target_pos' in self.params and 'target_time' in self.params:
            # guidance version
            self.locs = np.array(self.params['target_pos'])
            self.times = np.array(self.params['target_time'])
        else:
            raise NotImplementedError()
        # NOTE: assumes add_step will be called for initial state before model prediction as well
        self.times = self.times + 1
        self._agent_from_world = None
        self._per_step = []

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray):
        ''' stores all given positions for needed agents '''
        # which agent indices belong to each scene (list of np arrays with agent inds)
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_index"]), return_inverse=True)[1].numpy()
        scene_mask = local_scene_index == self.scene_idx
        pos_cur_scene = state_info["centroid"][scene_mask]
        if self._agent_from_world is None:
            # only want to record this at the first step (the planning step)
            #       b/c this is what the target is in.
            #       TODO: when the target is global, need to change this
            self._agent_from_world = state_info["agent_from_world"][scene_mask]
        pos_cur_scene = batch_nd_transform_points_np(pos_cur_scene[:,None], self._agent_from_world)[:,0]
        if self.agents is not None:
            pos_cur_scene = pos_cur_scene[self.agents]
        return pos_cur_scene

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        step_pos = self.compute_per_step(state_info, all_scene_index)
        self._per_step.append(step_pos)

    def get_episode_metrics(self):
        ''' finally see if were met at desired step'''
        all_pos = np.stack(self._per_step, axis=1)  # [num_agents, num_steps, 2]
        tgt_pos = all_pos[np.arange(all_pos.shape[0]),self.times]
        met = np.mean(np.linalg.norm(tgt_pos - self.locs, axis=-1))
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met
        return all_scene_met

class GlobalConstraintGuidance(GuidanceMetric):
    """
    How well constraint (waypoints at specific time) are met.
    """
    def reset(self):
        if 'locs' in self.params and 'times' in self.params:
            # true hard constraint
            self.locs = np.array(self.params['locs'])
            self.times = np.array(self.params['times'])
        elif 'target_pos' in self.params and 'target_time' in self.params:
            # guidance version
            self.locs = np.array(self.params['target_pos'])
            self.times = np.array(self.params['target_time'])
        else:
            raise NotImplementedError()
        # NOTE: assumes add_step will be called for initial state before model prediction as well
        self.times = self.times + 1
        self._per_step = []

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray):
        ''' stores all given positions for needed agents '''
        # which agent indices belong to each scene (list of np arrays with agent inds)
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_index"]), return_inverse=True)[1].numpy()
        scene_mask = local_scene_index == self.scene_idx
        pos_cur_scene = state_info["centroid"][scene_mask]
        if self.agents is not None:
            pos_cur_scene = pos_cur_scene[self.agents]
        return pos_cur_scene

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        step_pos = self.compute_per_step(state_info, all_scene_index)
        self._per_step.append(step_pos)

    def get_episode_metrics(self):
        ''' finally see if were met at desired step'''
        all_pos = np.stack(self._per_step, axis=1)  # [num_agents, num_steps, 2]
        tgt_pos = all_pos[np.arange(all_pos.shape[0]),self.times]
        met = np.mean(np.linalg.norm(tgt_pos - self.locs, axis=-1))
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met
        return all_scene_met

class MapCollisionGuidance(GuidanceMetric):
    """Compute the fraction of the time that the agent is in undrivable regions"""
    def reset(self):
        self._per_step = []

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray):
        # CHANGE: set ignore_if_unspecified to deal with a string type on scene_index
        obs = TensorUtils.to_tensor(state_info, ignore_if_unspecified=True)
        drivable_region = batch_utils().get_drivable_region_map(obs["image"])
        # print(obs["centroid"])
        # print(obs["raster_from_world"])
        centroid_raster = transform_points_tensor(obs["centroid"][:, None], obs["raster_from_world"])[:, 0]
        cur_yaw = transform_yaw(obs["yaw"], obs["raster_from_world"])[:,None] # have to use raster tf mat because the raster may not be up to date with the agent (i.e. the raster may be from an older frame)
        # cur_yaw = transform_yaw(obs["yaw"], obs["agent_from_world"])[:,None] # agent frame is same as raster, just scaled
        extent = obs["extent"][:,:2]
        # TODO: this is super hacky and assumes trajdata is being used.
        #       should really just transform corners in world frame then convert to raster.
        extent = get_raster_pix2m()*extent # convert to raster frame

        # filter for specified scene and agents only
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_index"]), return_inverse=True)[1].numpy()
        scene_mask = local_scene_index == self.scene_idx

        drivable_region = drivable_region[scene_mask]
        centroid_raster = centroid_raster[scene_mask]
        cur_yaw = cur_yaw[scene_mask]
        extent = extent[scene_mask]
        if self.agents is not None:
            drivable_region = drivable_region[self.agents]
            centroid_raster = centroid_raster[self.agents]
            cur_yaw = cur_yaw[self.agents]
            extent = extent[self.agents]

        off_road = Metrics.batch_detect_off_road_boxes(centroid_raster, cur_yaw, extent, drivable_region)
        # print(off_road)
        off_road = TensorUtils.to_numpy(off_road)

        return off_road

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        met = self.compute_per_step(state_info, all_scene_index)
        self._per_step.append(met)

    def get_episode_metrics(self):
        met_all_steps = np.stack(self._per_step, axis=1)
        # fraction of frames colliding
        # print(np.sum(met_all_steps, axis=1))
        met_guide_scene = np.sum(met_all_steps, axis=1) / float(met_all_steps.shape[1])
        met_guide_scene = np.mean(met_guide_scene)
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met_guide_scene
        return all_scene_met

class MapCollisionGuidanceDisk(GuidanceMetric):
    """Compute the fraction of the time that the agent is in undrivable regions.
            each agent is approximated by a single disk"""
    def reset(self):
        self._per_step = []

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray):
        # CHANGE: set ignore_if_unspecified to deal with a string type on scene_index
        obs = TensorUtils.to_tensor(state_info, ignore_if_unspecified=True)
        drivable_region = batch_utils().get_drivable_region_map(obs["image"])
        # print(obs["centroid"])
        # print(obs["raster_from_world"])
        centroid_raster = transform_points_tensor(obs["centroid"][:, None], obs["raster_from_world"])[:, 0]
        cur_yaw = transform_yaw(obs["yaw"], obs["raster_from_world"])[:,None] # have to use raster tf mat because the raster may not be up to date with the agent (i.e. the raster may be from an older frame)
        # cur_yaw = transform_yaw(obs["yaw"], obs["agent_from_world"])[:,None] # agent frame is same as raster, just scaled
        extent = obs["extent"][:,:2]
        extent = get_raster_pix2m()*extent # convert to raster frame

        # filter for specified scene and agents only
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_index"]), return_inverse=True)[1].numpy()
        scene_mask = local_scene_index == self.scene_idx

        drivable_region = drivable_region[scene_mask]
        centroid_raster = centroid_raster[scene_mask]
        cur_yaw = cur_yaw[scene_mask]
        extent = extent[scene_mask]
        if self.agents is not None:
            drivable_region = drivable_region[self.agents]
            centroid_raster = centroid_raster[self.agents]
            cur_yaw = cur_yaw[self.agents]
            extent = extent[self.agents]

        off_road = Metrics.batch_detect_off_road_disk(centroid_raster, extent, drivable_region)
        # print(off_road)
        off_road = TensorUtils.to_numpy(off_road)

        return off_road

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        met = self.compute_per_step(state_info, all_scene_index)
        self._per_step.append(met)

    def get_episode_metrics(self):
        met_all_steps = np.stack(self._per_step, axis=1)
        # fraction of frames colliding
        # print(np.sum(met_all_steps, axis=1))
        met_guide_scene = np.sum(met_all_steps, axis=1) / float(met_all_steps.shape[1])
        met_guide_scene = np.mean(met_guide_scene)
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met_guide_scene
        return all_scene_met

class SocialGroupGuidance(GuidanceMetric):
    """
    How well social group stays together.
    At each step computes how well min distance to another agent in the social group
    matches the desired social_dist.
    """
    def reset(self):
        self.social_dist = self.params['social_dist']
        self._per_step = []

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray):
        ''' stores all given positions for needed agents '''
        # which agent indices belong to each scene (list of np arrays with agent inds)
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_index"]), return_inverse=True)[1].numpy()
        scene_mask = local_scene_index == self.scene_idx
        pos_cur_scene = state_info["centroid"][scene_mask]

        if self.agents is not None:
            pos_cur_scene = pos_cur_scene[self.agents]

        B = pos_cur_scene.shape[0]
        pairwise_dist = np.linalg.norm(pos_cur_scene[np.newaxis] - pos_cur_scene[:,np.newaxis], axis=-1)
        pairwise_dist[np.arange(B),np.arange(B)] = np.inf
        min_dist = np.amin(pairwise_dist, axis=-1)
        step_err = np.abs(min_dist - self.social_dist)
        self._per_step.append(step_err)

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        self.compute_per_step(state_info, all_scene_index)

    def get_episode_metrics(self):
        all_err = np.stack(self._per_step, axis=1)  # [num_agents, num_steps]
        met = np.mean(all_err)
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met
        return all_scene_met


############## GUIDANCE METRICS utilities ########################

GUIDANCE_NAME_TO_METRICS = {
    'target_speed' :              {'target_speed' : TargetSpeedGuidance},
    'agent_collision' :           {'agent_collision' : AgentCollisionGuidance, 'agent_collision_disk' : AgentCollisionGuidanceDisk, 'social_dist' : SocialDistanceGuidance},
    'map_collision' :             {'map_collision' : MapCollisionGuidance, 'map_collision_disk' : MapCollisionGuidanceDisk},
    'target_pos_at_time' :        {'target_pos_at_time' : ConstraintGuidance},
    'target_pos' :                {'target_pos' : TargetPosGuidance},
    'global_target_pos_at_time' : {'global_target_pos_at_time' : GlobalConstraintGuidance},
    'global_target_pos' :         {'global_target_pos' : GlobalTargetPosGuidance},
    'social_group' :              {'social_group' : SocialGroupGuidance},
}

def guidance_metrics_from_config(guidance_config):
    '''
    Returns metrics objects to measure success of each guidance.
    '''
    metrics = dict()
    num_scenes = len(guidance_config)
    for si in range(num_scenes):
        cur_cfg_list = guidance_config[si]
        for ci, cur_cfg in enumerate(cur_cfg_list):
            guide_mets = GUIDANCE_NAME_TO_METRICS[cur_cfg['name']]
            for met_name, met_func in guide_mets.items():
                cur_metric = met_func(si, cur_cfg['params'], cur_cfg['agents'], num_scenes)
                # could be multiple of the same guidance in a scene
                metrics['guide_' + met_name + '_s%dg%d' % (si, ci)] = cur_metric
    return metrics

def constraint_metrics_from_config(constraint_config):
    metrics = dict()
    num_scenes = len(constraint_config)
    for si in range(num_scenes):
        cur_cfg = constraint_config[si]
        cur_metric = ConstraintGuidance(si,
                                        {'locs' : cur_cfg['locs'], 'times' : cur_cfg['times']},
                                        cur_cfg['agents'],
                                        num_scenes
                                        )
        metrics['guide_constraint_s%d' % (si)] = cur_metric
    return metrics
