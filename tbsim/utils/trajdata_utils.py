import torch
import numpy as np

import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.geometry_utils import transform_points_tensor
from tbsim.configs.base import ExperimentConfig

from trajdata import AgentType

BATCH_ENV = None
BATCH_RASTER_CFG = None

# need this for get_drivable_region_map
def set_global_trajdata_batch_env(batch_env):
    global BATCH_ENV
    BATCH_ENV = batch_env.split('-')[0] # if split is specified, remove it
# need this for rasterize_agents
def set_global_trajdata_batch_raster_cfg(raster_cfg):
    global BATCH_RASTER_CFG
    assert "include_hist" in raster_cfg
    assert "pixel_size" in raster_cfg
    assert "raster_size" in raster_cfg
    assert "ego_center" in raster_cfg
    assert "num_sem_layers" in raster_cfg
    assert "no_map_fill_value" in raster_cfg
    assert "drivable_layers" in raster_cfg
    BATCH_RASTER_CFG = raster_cfg

def get_raster_pix2m():
    return 1.0 / BATCH_RASTER_CFG["pixel_size"]

def trajdata2posyawspeed(state, nan_to_zero=True):
    """Converts trajdata's state format to pos, yaw, and speed. Set Nans to 0s"""
    
    if state.shape[-1] == 7:  # x, y, vx, vy, ax, ay, sin(heading), cos(heading)
        state = torch.cat((state[...,:6],torch.sin(state[...,6:7]),torch.cos(state[...,6:7])),-1)
    else:
        assert state.shape[-1] == 8
    pos = state[..., :2]
    yaw = torch.atan2(state[..., [-2]], state[..., [-1]])
    speed = torch.norm(state[..., 2:4], dim=-1)
    mask = torch.bitwise_not(torch.max(torch.isnan(state), dim=-1)[0])
    if nan_to_zero:
        pos[torch.bitwise_not(mask)] = 0.
        yaw[torch.bitwise_not(mask)] = 0.
        speed[torch.bitwise_not(mask)] = 0.
    return pos, yaw, speed, mask

def rasterize_agents(
        maps: torch.Tensor,
        agent_hist_pos: torch.Tensor,
        agent_hist_yaw: torch.Tensor,
        agent_mask: torch.Tensor,
        raster_from_agent: torch.Tensor,
        map_res: torch.Tensor,
) -> torch.Tensor:
    """Paint agent histories onto an agent-centric map image"""
    b, a, t, _ = agent_hist_pos.shape
    _, _, h, w = maps.shape
    maps = maps.clone()

    agent_hist_pos = agent_hist_pos.reshape(b, a * t, 2)
    raster_hist_pos = transform_points_tensor(agent_hist_pos, raster_from_agent)
    raster_hist_pos[~agent_mask.reshape(b, a * t)] = 0.0  # Set invalid positions to 0.0 Will correct below
    raster_hist_pos = raster_hist_pos.reshape(b, a, t, 2).permute(0, 2, 1, 3)  # [B, T, A, 2]
    raster_hist_pos[..., 0].clip_(0, (w - 1))
    raster_hist_pos[..., 1].clip_(0, (h - 1))
    raster_hist_pos = torch.round(raster_hist_pos).long()  # round pixels

    raster_hist_pos_flat = raster_hist_pos[..., 1] * w + raster_hist_pos[..., 0]  # [B, T, A]

    hist_image = torch.zeros(b, t, h * w, dtype=maps.dtype, device=maps.device)  # [B, T, H * W]

    hist_image.scatter_(dim=2, index=raster_hist_pos_flat[:, :, 1:], src=torch.ones_like(hist_image) * -1)  # mark other agents with -1
    hist_image.scatter_(dim=2, index=raster_hist_pos_flat[:, :, [0]], src=torch.ones_like(hist_image))  # mark ego with 1.
    hist_image[:, :, 0] = 0  # correct the 0th index from invalid positions
    hist_image[:, :, -1] = 0  # correct the maximum index caused by out of bound locations

    hist_image = hist_image.reshape(b, t, h, w)

    maps = torch.cat((hist_image, maps), dim=1)  # treat time as extra channels
    return maps


def get_drivable_region_map(maps):
    drivable_layers = BATCH_RASTER_CFG["drivable_layers"]
    if drivable_layers is None:
        # use defaults for known datasets
        env_name = BATCH_ENV
        if env_name in ['nusc_trainval', 'nusc_test', 'nusc_mini']:
            # first 3 layers are drivable
            # drivable_range = (-7, -4)
            drivable_layers = [0, 1, 2] #[-7, -6, -5]
        elif env_name in ['lyft_train', 'lyft_train_full', 'lyft_val', 'lyft_sample']:
            # drivable_range = (-3, -2)
            drivable_layers = [0] #[-3]
        elif env_name in ['orca_maps', 'orca_no_maps']: # if using a mixed dataset, orca_no_maps may have dummy map layers to parse
            # drivable_range = (-2, -1)
            drivable_layers = [0] #[-2]
        else:
            raise NotImplementedError("Must implement get_drivable_region_map for any new dataset from trajdata")

    drivable = None
    if len(drivable_layers) > 0:
        # convert to indices in the full rasterized stack of layers (which may include rasterized history)
        drivable_layers = -BATCH_RASTER_CFG["num_sem_layers"] + np.array(drivable_layers)
        if isinstance(maps, torch.Tensor):
            drivable = torch.amax(maps[..., drivable_layers, :, :], dim=-3)
            invalid_mask = ~compute_valid_map_mask(drivable.unsqueeze(1))
            # set batch indices with no map (infilled default value) to drivable by default for
            #       the sake of metrics
            drivable[invalid_mask] = 1.0
            drivable = drivable.bool()
        else:
            drivable = np.amax(maps[..., drivable_layers, :, :], axis=-3)
            invalid_mask = ~compute_valid_map_mask(drivable[:,np.newaxis])
            # set batch indices with no map (infilled default value) to drivable by default for
            #       the sake of metrics
            drivable[invalid_mask] = 1.0
            drivable = drivable.astype(np.bool)
    else:
        # the whole map is drivable
        if isinstance(maps, torch.Tensor):
            drivable_size = list(maps.size())
            drivable_size = drivable_size[:-3] + drivable_size[-2:]
            drivable = torch.ones(drivable_size, dtype=torch.bool).to(maps.device)
        else:
            drivable_size = list(maps.shape)
            drivable_size = drivable_size[:-3] + drivable_size[-2:]
            drivable = np.ones(drivable_size, dtype=bool)

    return drivable


def maybe_pad_neighbor(batch):
    """Pad neighboring agent's history to the same length as that of the ego using NaNs"""
    hist_len = batch["agent_hist"].shape[1]
    fut_len = batch["agent_fut"].shape[1]
    b, a, neigh_len, _ = batch["neigh_hist"].shape
    empty_neighbor = a == 0
    device = batch["neigh_hist"].device
    if empty_neighbor:
        batch["neigh_hist"] = torch.ones(b, 1, hist_len, batch["neigh_hist"].shape[-1]).to(device) * torch.nan
        batch["neigh_fut"] = torch.ones(b, 1, fut_len, batch["neigh_fut"].shape[-1]).to(device) * torch.nan
        batch["neigh_types"] = torch.zeros(b, 1).to(device)
        batch["neigh_hist_extents"] = torch.zeros(b, 1, hist_len, batch["neigh_hist_extents"].shape[-1]).to(device)
        batch["neigh_fut_extents"] = torch.zeros(b, 1, fut_len, batch["neigh_hist_extents"].shape[-1]).to(device)
    elif neigh_len < hist_len:
        hist_pad = torch.ones(b, a, hist_len - neigh_len, batch["neigh_hist"].shape[-1]).to(device) * torch.nan
        batch["neigh_hist"] = torch.cat((hist_pad, batch["neigh_hist"]), dim=2)
        hist_pad = torch.zeros(b, a, hist_len - neigh_len, batch["neigh_hist_extents"].shape[-1]).to(device)
        batch["neigh_hist_extents"] = torch.cat((hist_pad, batch["neigh_hist_extents"]), dim=2)


def parse_node_centric(batch: dict, overwrite_nan=True):
    maybe_pad_neighbor(batch)
    fut_pos, fut_yaw, _, fut_mask = trajdata2posyawspeed(batch["agent_fut"], nan_to_zero=overwrite_nan)
    hist_pos, hist_yaw, hist_speed, hist_mask = trajdata2posyawspeed(batch["agent_hist"], nan_to_zero=overwrite_nan)
    curr_speed = hist_speed[..., -1]
    curr_state = batch["curr_agent_state"]
    curr_yaw = curr_state[:, -1]
    curr_pos = curr_state[:, :2]

    # convert nuscenes types to l5kit types
    agent_type = batch["agent_type"]
    agent_type[agent_type < 0] = 0 # unknown
    agent_type[agent_type == 1] = 3 # vehicle
    agent_type[agent_type == 2] = 14 # pedestrian
    agent_type[agent_type == 3] = 10 # bicycle
    agent_type[agent_type == 4] = 11 # motorcycle
    # mask out invalid extents
    agent_hist_extent = batch["agent_hist_extent"]
    agent_hist_extent[torch.isnan(agent_hist_extent)] = 0.

    neigh_hist_pos, neigh_hist_yaw, neigh_hist_speed, neigh_hist_mask = trajdata2posyawspeed(batch["neigh_hist"], nan_to_zero=overwrite_nan)
    neigh_fut_pos, neigh_fut_yaw, _, neigh_fut_mask = trajdata2posyawspeed(batch["neigh_fut"], nan_to_zero=overwrite_nan)
    neigh_curr_speed = neigh_hist_speed[..., -1]
    neigh_types = batch["neigh_types"]
    # convert nuscenes types to l5kit types
    neigh_types[neigh_types < 0] = 0 # unknown
    neigh_types[neigh_types == 1] = 3 # vehicle
    neigh_types[neigh_types == 2] = 14 # pedestrian
    neigh_types[neigh_types == 3] = 10 # bicycle
    neigh_types[neigh_types == 4] = 11 # motorcycle
    # mask out invalid extents
    neigh_hist_extents = batch["neigh_hist_extents"]
    neigh_hist_extents[torch.isnan(neigh_hist_extents)] = 0.

    world_from_agents = torch.inverse(batch["agents_from_world_tf"])

    raster_cfg = BATCH_RASTER_CFG
    map_res = 1.0 / raster_cfg["pixel_size"] # convert to pixels/meter
    h = w = raster_cfg["raster_size"]
    ego_cent = raster_cfg["ego_center"]

    raster_from_agent = torch.Tensor([
            [map_res, 0, ((1.0 + ego_cent[0])/2.0) * w],
            [0, map_res, ((1.0 + ego_cent[1])/2.0) * h],
            [0, 0, 1]
    ]).to(curr_state.device)
    
    bsize = batch["agents_from_world_tf"].shape[0]
    agent_from_raster = torch.inverse(raster_from_agent)
    raster_from_agent = TensorUtils.unsqueeze_expand_at(raster_from_agent, size=bsize, dim=0)
    agent_from_raster = TensorUtils.unsqueeze_expand_at(agent_from_raster, size=bsize, dim=0)
    raster_from_world = torch.bmm(raster_from_agent, batch["agents_from_world_tf"])

    all_hist_pos = torch.cat((hist_pos[:, None], neigh_hist_pos.to(hist_pos.device)), dim=1)
    all_hist_yaw = torch.cat((hist_yaw[:, None], neigh_hist_yaw.to(hist_pos.device)), dim=1)
    all_hist_mask = torch.cat((hist_mask[:, None], neigh_hist_mask.to(hist_pos.device)), dim=1)

    maps_rasterize_in = batch["maps"]
    if maps_rasterize_in is None and BATCH_RASTER_CFG["include_hist"]:
        maps_rasterize_in = torch.empty((bsize, 0, h, w)).to(all_hist_pos.device)
    elif maps_rasterize_in is not None:
        maps_rasterize_in = verify_map(maps_rasterize_in)

    if BATCH_RASTER_CFG["include_hist"]:
        # first T channels are rasterized history (single pixel where agent is)
        #       -1 for ego, 1 for others
        # last num_sem_layers are direclty the channels from data loader
        maps = rasterize_agents(
            maps_rasterize_in,
            all_hist_pos,
            all_hist_yaw,
            all_hist_mask,
            raster_from_agent,
            map_res
        )
    else:
        maps = maps_rasterize_in

    drivable_map = None
    if batch["maps"] is not None:
        drivable_map = get_drivable_region_map(maps_rasterize_in)

    extent_scale = 1.0
    d = dict(
        image=maps,
        drivable_map=drivable_map,
        target_positions=fut_pos,
        target_yaws=fut_yaw,
        target_availabilities=fut_mask,
        history_positions=hist_pos,
        history_yaws=hist_yaw,
        history_speeds=hist_speed,
        history_availabilities=hist_mask,
        curr_speed=curr_speed,
        centroid=curr_pos,
        yaw=curr_yaw,
        type=agent_type,
        extent=agent_hist_extent.max(dim=-2)[0] * extent_scale,
        raster_from_agent=raster_from_agent,
        agent_from_raster=agent_from_raster,
        raster_from_world=raster_from_world,
        agent_from_world=batch["agents_from_world_tf"],
        world_from_agent=world_from_agents,
        all_other_agents_history_positions=neigh_hist_pos,
        all_other_agents_history_yaws=neigh_hist_yaw,
        all_other_agents_history_speeds=neigh_hist_speed,
        all_other_agents_history_availabilities=neigh_hist_mask,
        all_other_agents_history_availability=neigh_hist_mask,  # dump hack to agree with l5kit's typo ...
        all_other_agents_curr_speed=neigh_curr_speed,
        all_other_agents_future_positions=neigh_fut_pos,
        all_other_agents_future_yaws=neigh_fut_yaw,
        all_other_agents_future_availability=neigh_fut_mask,
        all_other_agents_types=neigh_types,
        all_other_agents_extents=neigh_hist_extents.max(dim=-2)[0] * extent_scale,
        all_other_agents_history_extents=neigh_hist_extents * extent_scale,
    )
    if "agent_lanes" in batch:
        d["ego_lanes"] = batch["agent_lanes"]
    return d

def verify_map(batch_maps):
    '''
    Verify and expand map to the number of expected channels if necessary.
    '''
    # if we use incl_map with trajdata, but the data does not contain a map, it will
    #       return 1 empty channel. Need to expand to the expected size given in config.
    if isinstance(batch_maps, torch.Tensor):
        if batch_maps.size(1) != BATCH_RASTER_CFG["num_sem_layers"]:
            assert batch_maps.size(1) == 1, "maps from trajdata have an unexpected number of layers"
            batch_maps = batch_maps.expand(-1, BATCH_RASTER_CFG["num_sem_layers"], -1, -1)
    else:
        if batch_maps.shape[1] != BATCH_RASTER_CFG["num_sem_layers"]:
            assert batch_maps.shape[1] == 1, "maps from trajdata have an unexpected number of layers"
            batch_maps = np.repeat(batch_maps, BATCH_RASTER_CFG["num_sem_layers"], axis=1)

    return batch_maps

def compute_valid_map_mask(batch_maps):
    '''
     - batch_maps (B, C, H, W)
    '''
    if isinstance(batch_maps, torch.Tensor):
        _, C, H, W = batch_maps.size()
        map_valid_mask = ~(torch.sum(torch.isclose(batch_maps, torch.tensor([BATCH_RASTER_CFG["no_map_fill_value"]], device=batch_maps.device)), dim=[1,2,3]) == C*H*W)
    else:
        B, C, H, W = batch_maps.shape
        map_valid_mask = ~(np.sum(np.isclose(batch_maps, np.array([BATCH_RASTER_CFG["no_map_fill_value"]])).reshape((B,-1)), axis=1) == C*H*W)
    return map_valid_mask

@torch.no_grad()
def parse_trajdata_batch(batch: dict, overwrite_nan=True):
    
    if "num_agents" in batch:
        # scene centric
        raise NotImplementedError("Currently only support agent-centric trajdata, not scene-centric")        
    else:
        # agent centric
        d = parse_node_centric(batch, overwrite_nan=overwrite_nan)

    batch = dict(batch)
    batch.update(d)
    if overwrite_nan:
        for k,v in batch.items():
            if isinstance(v,torch.Tensor):
                batch[k]=v.nan_to_num(0)
    batch.pop("agent_name", None)
    batch.pop("robot_fut", None)
    batch.pop("scene_ids", None)
    return batch

TRAJDATA_AGENT_TYPE_MAP = {
    'unknown' : AgentType.UNKNOWN, 
    'vehicle' : AgentType.VEHICLE,
    'pedestrian' : AgentType.PEDESTRIAN,
    'bicycle' : AgentType.BICYCLE,
    'motorcycle' : AgentType.MOTORCYCLE
}

def get_modality_shapes(cfg: ExperimentConfig):
    assert "num_sem_layers" in cfg.env.rasterizer.keys(), "must indicate number of semantic layer in env config for trajdata"
    num_sem_layers = cfg.env.rasterizer.num_sem_layers
    hist_layer_size = (cfg.algo.history_num_frames + 1) if cfg.env.rasterizer.include_hist else 0
    num_channels = hist_layer_size + num_sem_layers
    h = cfg.env.rasterizer.raster_size
    return dict(image=(num_channels, h, h))