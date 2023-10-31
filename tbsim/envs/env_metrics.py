import abc
import numpy as np
from typing import Dict
import math

import torch

import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.batch_utils import batch_utils
from tbsim.utils.geometry_utils import transform_points_tensor, transform_yaw, detect_collision, CollisionType
from tbsim.utils.trajdata_utils import get_raster_pix2m
import tbsim.utils.metrics as Metrics
from collections import defaultdict
import pandas as pd

class EnvMetrics(abc.ABC):
    def __init__(self):
        self._df = None
        self._scene_ts = defaultdict(lambda:0)
        self.reset()

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def add_step(self, state_info: Dict, all_scene_index: np.ndarray):
        pass

    @abc.abstractmethod
    def get_episode_metrics(self) -> Dict[str, np.ndarray]:
        pass

    def get_multi_episode_metrics(self) -> Dict[str, np.ndarray]:
        pass
    
    def multi_episode_reset(self):
        pass

    def __len__(self):
        return max(self._scene_ts.values()) if len(self._scene_ts)>0 else 0


def step_aggregate_per_scene(agent_met, agent_scene_index, all_scene_index, agg_func=np.mean):
    """
    Aggregate per-step metrics for each scene.

    1. if there are more than one agent per scene, aggregate their metrics for each scene using @agg_func.
    2. if there are zero agent per scene, the returned mask should have 0 for that scene

    Args:
        agent_met (np.ndarray): metrics for all agents and scene [num_agents, ...]
        agent_scene_index (np.ndarray): scene index for each agent [num_agents]
        all_scene_index (list, np.ndarray): a list of scene indices [num_scene]
        agg_func: function to aggregate metrics value across all agents in a scene

    Returns:
        met_per_scene (np.ndarray): [num_scene]
        met_per_scene_mask (np.ndarray): [num_scene]
    """
    met_per_scene = split_agents_by_scene(agent_met, agent_scene_index, all_scene_index)
    met_agg_per_scene = []
    for met in met_per_scene:
        if len(met) > 0:
            met_agg_per_scene.append(agg_func(met))
        else:
            met_agg_per_scene.append(np.zeros_like(agent_met[0]))
    met_mask_per_scene = [len(met) > 0 for met in met_per_scene]
    return np.stack(met_agg_per_scene, axis=0), np.array(met_mask_per_scene)


def split_agents_by_scene(agent, agent_scene_index, all_scene_index):

    assert agent.shape[0] == agent_scene_index.shape[0]
    agent_split = []
    for si in all_scene_index:
        agent_split.append(agent[agent_scene_index == si])
    return agent_split


def agent_index_by_scene(agent_scene_index, all_scene_index):
    agent_split = []
    for si in all_scene_index:
        agent_split.append(np.where(agent_scene_index == si)[0])
    return agent_split

class OffRoadRate(EnvMetrics):
    """Compute the fraction of the time that the agent is in undrivable regions"""
    def reset(self):
        self._df = pd.DataFrame(columns = ['scene_index', 'track_id', 'ts', "met"])
        self._scene_ts = defaultdict(lambda:0)

    @staticmethod
    def compute_per_step(state_info: dict, all_scene_index: np.ndarray):
        # CHANGE: set ignore_if_unspecified to deal with a string type on scene_index
        obs = TensorUtils.to_tensor(state_info, ignore_if_unspecified=True)
        drivable_region = batch_utils().get_drivable_region_map(obs["image"])
        # print(obs["centroid"])
        # print(obs["raster_from_world"])
        centroid_raster = transform_points_tensor(obs["centroid"][:, None], obs["raster_from_world"])[:, 0]
        # cur_yaw = transform_yaw(obs["yaw"], obs["agent_from_world"])[:,None] # agent frame is same as raster, just scaled
        cur_yaw = transform_yaw(obs["yaw"], obs["raster_from_world"])[:,None] # have to use raster tf mat because the raster may not be up to date with the agent (i.e. the raster may be from an older frame)
        extent = obs["extent"][:,:2]
        extent = get_raster_pix2m()*extent # convert to raster frame

        off_road_out = np.ones((centroid_raster.size(0))) * np.nan

        valid_mask = torch.sum(torch.isnan(centroid_raster), dim=-1) == 0
        if torch.sum(valid_mask) == 0:
            return off_road_out
        off_road = Metrics.batch_detect_off_road_boxes(centroid_raster[valid_mask],
                                                        cur_yaw[valid_mask],
                                                        extent[valid_mask],
                                                        drivable_region[valid_mask])
        # print(off_road)
        off_road = TensorUtils.to_numpy(off_road)
        off_road_out[valid_mask.cpu().numpy()] = off_road
        return off_road_out

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        met = self.compute_per_step(state_info, all_scene_index)
        
        ts = np.array([self._scene_ts[sid] for sid in state_info["scene_index"]])
        step_df = dict(scene_index=state_info["scene_index"],
                       track_id=state_info["track_id"],
                       ts=ts,
                       met=met)
        step_df = pd.DataFrame(step_df)
        self._df = pd.concat((self._df,step_df))
        for sid in np.unique(state_info["scene_index"]):
            self._scene_ts[sid]+=1

    def get_episode_metrics(self):
        # df sums and means automatically ignore nan
        self._df.set_index(["scene_index","track_id","ts"])
        metric_by_agt = self._df.groupby(["scene_index","track_id"])["met"].sum()
        metric_nframe = metric_by_agt.groupby(["scene_index"]).mean().to_numpy()
        metric_rate = self._df.groupby(["scene_index"])["met"].mean().to_numpy()
        return {
            "rate" : metric_rate,
            "nframe" : metric_nframe
        }

class DiskOffRoadRate(EnvMetrics):
    """Compute the fraction of the time that the agent is in undrivable regions"""
    def reset(self):
        self._df = pd.DataFrame(columns = ['scene_index', 'track_id', 'ts', "met"])
        self._scene_ts = defaultdict(lambda:0)

    @staticmethod
    def compute_per_step(state_info: dict, all_scene_index: np.ndarray):
        # CHANGE: set ignore_if_unspecified to deal with a string type on scene_index
        obs = TensorUtils.to_tensor(state_info, ignore_if_unspecified=True)
        drivable_region = batch_utils().get_drivable_region_map(obs["image"])
        # print(obs["centroid"])
        # print(obs["raster_from_world"])
        centroid_raster = transform_points_tensor(obs["centroid"][:, None], obs["raster_from_world"])[:, 0]
        # cur_yaw = transform_yaw(obs["yaw"], obs["agent_from_world"])[:,None] # agent frame is same as raster, just scaled
        cur_yaw = transform_yaw(obs["yaw"], obs["raster_from_world"])[:,None] # have to use raster tf mat because the raster may not be up to date with the agent (i.e. the raster may be from an older frame)
        extent = obs["extent"][:,:2]
        extent = get_raster_pix2m()*extent # convert to raster frame

        off_road_out = np.ones((centroid_raster.size(0))) * np.nan

        valid_mask = torch.sum(torch.isnan(centroid_raster), dim=-1) == 0
        if torch.sum(valid_mask) == 0:
            return off_road_out
        # off_road = Metrics.batch_detect_off_road_boxes(centroid_raster[valid_mask],
        #                                                 cur_yaw[valid_mask],
        #                                                 extent[valid_mask],
        #                                                 drivable_region[valid_mask])
        off_road = Metrics.batch_detect_off_road_disk(centroid_raster[valid_mask],
                                                        extent[valid_mask],
                                                        drivable_region[valid_mask])
        # print(off_road)
        off_road = TensorUtils.to_numpy(off_road)
        off_road_out[valid_mask.cpu().numpy()] = off_road
        return off_road_out

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        met = self.compute_per_step(state_info, all_scene_index)
        
        ts = np.array([self._scene_ts[sid] for sid in state_info["scene_index"]])
        step_df = dict(scene_index=state_info["scene_index"],
                       track_id=state_info["track_id"],
                       ts=ts,
                       met=met)
        step_df = pd.DataFrame(step_df)
        self._df = pd.concat((self._df,step_df))
        for sid in np.unique(state_info["scene_index"]):
            self._scene_ts[sid]+=1

    def get_episode_metrics(self):
        # df sums and means automatically ignore nan
        self._df.set_index(["scene_index","track_id","ts"])
        metric_by_agt = self._df.groupby(["scene_index","track_id"])["met"].sum()
        metric_nframe = metric_by_agt.groupby(["scene_index"]).mean().to_numpy()
        metric_rate = self._df.groupby(["scene_index"])["met"].mean().to_numpy()
        return {
            "rate" : metric_rate,
            "nframe" : metric_nframe
        }

class SemLayerRate(EnvMetrics):
    """
    Compute the fraction of the time that the agent centroid is on each layer of the semantic map
    Note: assumes same agent/scenes over time.
    """
    def reset(self):
        # self._df = pd.DataFrame(columns = ['scene_index', 'track_id', 'ts', "met"])
        # self._scene_ts = defaultdict(lambda:0)
        self._per_step = []
        self._scene_masks = None

    @staticmethod
    def compute_per_step(state_info: dict, all_scene_index: np.ndarray):
        obs = TensorUtils.to_tensor(state_info, ignore_if_unspecified=True)

        sem_map = obs["image"]
        num_layers = sem_map.size(1)

        on_layers = []
        for li in range(num_layers+1):
            # print(obs["centroid"])
            # print(obs["raster_from_world"])
            centroid_raster = transform_points_tensor(obs["centroid"][:, None], obs["raster_from_world"])[:, 0]

            cur_on_layer_out = np.ones((centroid_raster.size(0))) * np.nan
            valid_mask = torch.sum(torch.isnan(centroid_raster), dim=-1) == 0
            if torch.sum(valid_mask) == 0:
                on_layers.append(cur_on_layer_out)
                continue

            if li == num_layers:
                # check if on no layers
                cur_layer_map = ~(torch.amax(sem_map[valid_mask], dim=-3).bool())
            else:
                cur_layer_map = sem_map[valid_mask][:,li].bool()
            cur_on_layer = 1.0 - Metrics.batch_detect_off_road(centroid_raster[valid_mask], cur_layer_map)  # [num_agents]
            # print(off_road)
            cur_on_layer = TensorUtils.to_numpy(cur_on_layer)
            cur_on_layer_out[valid_mask.cpu().numpy()] = cur_on_layer
            on_layers.append(cur_on_layer_out)

        on_layers = np.stack(on_layers, axis=1) # num_agents x num_layers

        return on_layers

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        met = self.compute_per_step(state_info, all_scene_index)
        self._per_step.append(met)
        if self._scene_masks is None:
            agent_scene_index = state_info["scene_index"]
            self._scene_masks = agent_index_by_scene(agent_scene_index, all_scene_index)

    def get_episode_metrics(self):
        met_all_steps = np.stack(self._per_step, axis=1) # num_agents x T x num_layers
        met_per_scene = []
        for scene_mask in self._scene_masks:
            cur_scene_met = met_all_steps[scene_mask]
            sem_rate = np.nanmean(cur_scene_met, axis=1) # num_agents x num_layers
            sem_scene = np.mean(sem_rate, axis=0) # num_layers
            met_per_scene.append(sem_scene)
        # track each layer separately
        num_layers = len(met_per_scene[0]) - 1
        met_dict = dict()
        for li in range(num_layers+1):
            layer_met = [smet[li] for smet in met_per_scene]
            layer_str = "no_layer" if li == num_layers else "layer%02d" % (li)
            met_dict[layer_str] = np.array(layer_met)
        return met_dict

class CollisionRate(EnvMetrics):
    """Compute collision rate across all agents in a batch of data."""
    def __init__(self):
        super(CollisionRate, self).__init__()
        self._df = pd.DataFrame(columns = ['scene_index', 'track_id', 'ts', 'type', "met"])
        self._scene_ts = defaultdict(lambda:0)

    def reset(self):
        self._df = pd.DataFrame(columns = ['scene_index', 'track_id', 'ts', 'type', "met"])
        self._scene_ts = defaultdict(lambda:0)

    @staticmethod
    def compute_per_step(state_info: dict, all_scene_index: np.ndarray):
        """Compute per-agent and per-scene collision rate and type"""
        agent_scene_index = state_info["scene_index"]
        pos_per_scene = split_agents_by_scene(state_info["centroid"], agent_scene_index, all_scene_index)
        yaw_per_scene = split_agents_by_scene(state_info["yaw"], agent_scene_index, all_scene_index)
        extent_per_scene = split_agents_by_scene(state_info["extent"][..., :2], agent_scene_index, all_scene_index)
        agent_index_per_scene = agent_index_by_scene(agent_scene_index, all_scene_index)

        num_scenes = len(all_scene_index)
        num_agents = len(agent_scene_index)

        coll_rates = dict()
        for k in CollisionType:
            coll_rates[k] = np.zeros(num_agents)
        coll_rates["coll_any"] = np.zeros(num_agents)

        # for each scene, compute collision rate
        for i in range(num_scenes):
            num_agents_in_scene = pos_per_scene[i].shape[0]
            for j in range(num_agents_in_scene):
                other_agent_mask = np.arange(num_agents_in_scene) != j
                valid_mask = np.logical_not(np.sum(np.isnan(pos_per_scene[i]), axis=-1) > 0)
                if not valid_mask[j]:
                    continue
                other_agent_mask = np.logical_and(other_agent_mask, valid_mask)
                if np.sum(other_agent_mask) == 0:
                    continue
                coll = detect_collision(
                    ego_pos=pos_per_scene[i][j],
                    ego_yaw=yaw_per_scene[i][j],
                    ego_extent=extent_per_scene[i][j],
                    other_pos=pos_per_scene[i][other_agent_mask],
                    other_yaw=yaw_per_scene[i][other_agent_mask],
                    other_extent=extent_per_scene[i][other_agent_mask]
                )
                
                if coll is not None:
                    coll_rates[coll[0]][agent_index_per_scene[i][j]] = 1.
                    coll_rates["coll_any"][agent_index_per_scene[i][j]] = 1.

        # compute per-scene collision counts (for visualization purposes)
        coll_counts = dict()
        for k in coll_rates:
            coll_counts[k], _ = step_aggregate_per_scene(
                coll_rates[k],
                agent_scene_index,
                all_scene_index,
                agg_func=np.sum
            )

        return coll_rates, coll_counts

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        
        met_all, _ = self.compute_per_step(state_info, all_scene_index)
        ts = np.array([self._scene_ts[sid] for sid in state_info["scene_index"]])
        step_df = []
        for k in met_all:
            if k=="coll_any":
                type=-1
            else:
                type=k
            step_df_k = dict(scene_index=state_info["scene_index"],
                        track_id=state_info["track_id"],
                        ts=ts,
                        type=type,
                        met=met_all[k])
            step_df.append(pd.DataFrame(step_df_k))
        step_df = pd.concat(step_df)
        self._df = pd.concat((self._df,step_df))
        for sid in np.unique(state_info["scene_index"]):
            self._scene_ts[sid]+=1

    def get_episode_metrics(self):

        self._df.set_index(["scene_index","track_id","type","ts"])
        coll_whole_horizon = self._df.groupby(["scene_index","track_id","type"])["met"].max()
        met_all = dict()
        
        for k in CollisionType:
            coll_data = coll_whole_horizon[coll_whole_horizon.index.isin([k],level=2)]
            met_all[str(k)] = coll_data.groupby(["scene_index"]).mean().to_numpy()
        coll_data = coll_whole_horizon[coll_whole_horizon.index.isin([-1],level=2)]
        met_all["coll_any"] = coll_data.groupby(["scene_index"]).mean().to_numpy()
        return met_all

class DiskCollisionRate(EnvMetrics):
    """Compute collision rate across all agents in a batch of data."""
    def __init__(self):
        super(DiskCollisionRate, self).__init__()
        self._df = pd.DataFrame(columns = ['scene_index', 'track_id', 'ts', 'type', "met"])
        self._scene_ts = defaultdict(lambda:0)

    def reset(self):
        self._df = pd.DataFrame(columns = ['scene_index', 'track_id', 'ts', 'type', "met"])
        self._scene_ts = defaultdict(lambda:0)

    @staticmethod
    def compute_per_step(state_info: dict, all_scene_index: np.ndarray):
        """Compute per-agent and per-scene collision rate and type"""
        agent_scene_index = state_info["scene_index"]
        pos_per_scene = split_agents_by_scene(state_info["centroid"], agent_scene_index, all_scene_index)
        yaw_per_scene = split_agents_by_scene(state_info["yaw"], agent_scene_index, all_scene_index)
        extent_per_scene = split_agents_by_scene(state_info["extent"][..., :2], agent_scene_index, all_scene_index)
        agent_index_per_scene = agent_index_by_scene(agent_scene_index, all_scene_index)

        num_scenes = len(all_scene_index)
        num_agents = len(agent_scene_index)

        coll_rates = dict()
        for k in CollisionType:
            coll_rates[k] = np.zeros(num_agents)
        coll_rates["coll_any"] = np.zeros(num_agents)

        # for each scene, compute collision rate
        for i in range(num_scenes):
            num_agents_in_scene = pos_per_scene[i].shape[0]
            rad_cur_scene = np.nanmin(extent_per_scene[i], axis=-1) / 2.0
            for j in range(num_agents_in_scene):
                other_agent_mask = np.arange(num_agents_in_scene) != j
                valid_mask = np.logical_not(np.sum(np.isnan(pos_per_scene[i]), axis=-1) > 0)
                if not valid_mask[j]:
                    continue
                other_agent_mask = np.logical_and(other_agent_mask, valid_mask)
                if np.sum(other_agent_mask) == 0:
                    continue

                neighbor_dist = np.linalg.norm(pos_per_scene[i][j:j+1] - pos_per_scene[i][other_agent_mask], axis=-1)
                min_allowed_dist = rad_cur_scene[j] + rad_cur_scene[other_agent_mask]
                coll = np.sum(neighbor_dist < min_allowed_dist) > 0
                if coll:
                    coll_rates["coll_any"][agent_index_per_scene[i][j]] = 1.

        # compute per-scene collision counts (for visualization purposes)
        coll_counts = dict()
        for k in coll_rates:
            coll_counts[k], _ = step_aggregate_per_scene(
                coll_rates[k],
                agent_scene_index,
                all_scene_index,
                agg_func=np.sum
            )

        return coll_rates, coll_counts

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        
        met_all, _ = self.compute_per_step(state_info, all_scene_index)
        ts = np.array([self._scene_ts[sid] for sid in state_info["scene_index"]])
        step_df = []
        for k in met_all:
            if k=="coll_any":
                type=-1
            else:
                type=k
            step_df_k = dict(scene_index=state_info["scene_index"],
                        track_id=state_info["track_id"],
                        ts=ts,
                        type=type,
                        met=met_all[k])
            step_df.append(pd.DataFrame(step_df_k))
        step_df = pd.concat(step_df)
        self._df = pd.concat((self._df,step_df))
        for sid in np.unique(state_info["scene_index"]):
            self._scene_ts[sid]+=1

    def get_episode_metrics(self):

        self._df.set_index(["scene_index","track_id","type","ts"])
        coll_whole_horizon = self._df.groupby(["scene_index","track_id","type"])["met"].max()
        met_all = dict()
        
        for k in CollisionType:
            coll_data = coll_whole_horizon[coll_whole_horizon.index.isin([k],level=2)]
            met_all[str(k)] = coll_data.groupby(["scene_index"]).mean().to_numpy()
        coll_data = coll_whole_horizon[coll_whole_horizon.index.isin([-1],level=2)]
        met_all["coll_any"] = coll_data.groupby(["scene_index"]).mean().to_numpy()
        return met_all


class CriticalFailure(EnvMetrics):
    """Metrics that report failures caused by either collision or offroad"""
    def __init__(self, num_collision_frames=1, num_offroad_frames=3):
        super(CriticalFailure, self).__init__()
        self._df = pd.DataFrame(columns=["scene_index","track_id","ts","offroad","collision"])
        self._scene_ts = defaultdict(lambda:0)

    def reset(self):
        self._df = pd.DataFrame(columns=["scene_index","track_id","ts","offroad","collision"])
        self._scene_ts = defaultdict(lambda:0)


    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        met_all = dict(
            offroad=OffRoadRate.compute_per_step(state_info, all_scene_index),
            collision=CollisionRate.compute_per_step(state_info, all_scene_index)[0]["coll_any"]
        )
        ts = np.array([self._scene_ts[sid] for sid in state_info["scene_index"]])
        step_df = dict(scene_index=state_info["scene_index"],
                       track_id=state_info["track_id"],
                       ts=ts,
                       offroad=met_all["offroad"],
                       collision = met_all["collision"])
        step_df = pd.DataFrame(step_df)
        self._df = pd.concat((self._df,step_df))
        for sid in np.unique(state_info["scene_index"]):
            self._scene_ts[sid]+=1
    
    def get_per_agent_metrics(self):
        coll_fail_cases = self._df.groupby(["scene_index","track_id"])["collision"].any()
        offroad_fail_cases = self._df.groupby(["scene_index","track_id"])["offroad"].any()
        any_fail_cases = coll_fail_cases|offroad_fail_cases
        return dict(offroad=offroad_fail_cases,collision=coll_fail_cases,any=any_fail_cases)


    def get_episode_metrics(self) -> Dict[str, np.ndarray]:
        num_steps = len(self)
        grid_points = np.arange(5,num_steps,5)

        coll_fail_cases = self._df.groupby(["scene_index","track_id"])["collision"].any()
        coll_by_scene = coll_fail_cases.groupby(["scene_index"])
        coll_fail_rate = (coll_by_scene.sum()/coll_by_scene.count()).to_numpy()
        offroad_fail_cases = self._df.groupby(["scene_index","track_id"])["offroad"].any()
        offroad_by_scene = offroad_fail_cases.groupby(["scene_index"])
        offroad_fail_rate = (offroad_by_scene.sum()/offroad_by_scene.count()).to_numpy()
        any_fail_cases = coll_fail_cases | offroad_fail_cases
        any_fail_by_scene = any_fail_cases.groupby(["scene_index"])
        any_fail_rate = (any_fail_by_scene.sum()/any_fail_by_scene.count()).to_numpy()

        met = dict(failure_offroad=offroad_fail_rate,failure_collision=coll_fail_rate,failure_any=any_fail_rate)
        for t in grid_points:
            df_sel = self._df.loc[self._df["ts"]<t]
            coll_fail_cases = df_sel.groupby(["scene_index","track_id"])["collision"].any()
            coll_by_scene = coll_fail_cases.groupby(["scene_index"])
            coll_fail_rate = (coll_by_scene.sum()/coll_by_scene.count()).to_numpy()
            offroad_fail_cases = df_sel.groupby(["scene_index","track_id"])["offroad"].any()
            offroad_by_scene = offroad_fail_cases.groupby(["scene_index"])
            offroad_fail_rate = (offroad_by_scene.sum()/offroad_by_scene.count()).to_numpy()
            any_fail_cases = coll_fail_cases | offroad_fail_cases
            any_fail_by_scene = any_fail_cases.groupby(["scene_index"])
            any_fail_rate = (any_fail_by_scene.sum()/any_fail_by_scene.count()).to_numpy()
            met["failure_offroad@{}".format(t)]=offroad_fail_rate
            met["failure_collision@{}".format(t)]=coll_fail_rate
            met["failure_any@{}".format(t)]=any_fail_rate
        return met

class Comfort(EnvMetrics):
    # compute stats at lower res than actual sim b/c for GT
    #       they are usually linearly inteprolated from low framerate data
    def __init__(self, sim_dt=0.1, stat_dt=0.5):
        super(Comfort, self).__init__()
        self.sim_dt = sim_dt
        self.stat_dt = stat_dt
        assert stat_dt >= sim_dt
        
    """Compute metrics relevant to comfort (speed, accels, jerk)"""
    def reset(self):
        self._per_step = []
        self._scene_masks = None

    @staticmethod
    def compute_per_step(state_info: dict, all_scene_index: np.ndarray):
        cur_state = np.concatenate([state_info["centroid"], state_info["yaw"][:,np.newaxis]], axis=1)
        return cur_state

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        met = self.compute_per_step(state_info, all_scene_index)
        self._per_step.append(met)
        if self._scene_masks is None:
            agent_scene_index = state_info["scene_index"]
            self._scene_masks = agent_index_by_scene(agent_scene_index, all_scene_index)

    def get_episode_metrics(self):
        state_traj = np.stack(self._per_step, axis=1) # num_agents x T x 2
        # downsample
        dt_ratio = int(math.ceil(self.stat_dt / self.sim_dt))
        state_traj = state_traj[:,::dt_ratio]

        pos_traj = state_traj[..., :2]
        yaw_traj = state_traj[..., 2]
        vel_traj = np.diff(pos_traj, axis=1) / self.stat_dt
        speed = np.linalg.norm(vel_traj, axis=-1)

        accel_traj = np.diff(vel_traj, axis=1) / self.stat_dt
        accel_norm = np.linalg.norm(accel_traj, axis=-1)
        # magnitude of accel, not the direction
        lon_acc = np.abs(accel_norm * np.cos(yaw_traj[:,:accel_norm.shape[1]]))
        lat_acc = np.abs(accel_norm * np.sin(yaw_traj[:,:accel_norm.shape[1]]))
        jerk = np.abs(np.diff(accel_norm, axis=1) / self.stat_dt)

        # nanmean over each agent first, then group by scene
        speed = np.nanmean(speed, axis=1)
        lon_acc = np.nanmean(lon_acc, axis=1)
        lat_acc = np.nanmean(lat_acc, axis=1)
        jerk = np.nanmean(jerk, axis=1)

        met_per_scene = {
            'speed' : [],
            'lon_acc' : [],
            'lat_acc' : [],
            'jerk' : []
        }
        for scene_mask in self._scene_masks:
            met_per_scene['speed'].append(np.nanmean(speed[scene_mask]))
            met_per_scene['lon_acc'].append(np.nanmean(lon_acc[scene_mask]))
            met_per_scene['lat_acc'].append(np.nanmean(lat_acc[scene_mask]))
            met_per_scene['jerk'].append(np.nanmean(jerk[scene_mask]))

        for k,v in met_per_scene.items():
            met_per_scene[k] = np.array(v)

        return met_per_scene
