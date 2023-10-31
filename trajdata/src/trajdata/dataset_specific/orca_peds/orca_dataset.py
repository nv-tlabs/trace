from pathlib import Path
from typing import Dict, Final, List, Optional, Tuple, Type
from random import Random
import pickle
from collections import OrderedDict
from tqdm import tqdm

import numpy as np
import pandas as pd
import cv2

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures.agent import AgentMetadata, AgentType, FixedExtent
from trajdata.data_structures.environment import EnvMetadata
from trajdata.data_structures.map import MapMetadata
from trajdata.data_structures.scene_metadata import Scene, SceneMetadata
from trajdata.data_structures.scene_tag import SceneTag
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.dataset_specific.scene_records import OrcaRecord
from trajdata.utils import arr_utils

ORCA_DT: Final[float] = 0.1

ORCA_NUM_SCENES: Final[int] = 1000
ALL_SCENES = ['scene_%06d' % (sidx) for sidx in range(ORCA_NUM_SCENES)]
ORCA_MAP_DIM = (90.0, 90.0) # meters, conservative range that covers all trajectories

class OrcaDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
         # both "orca_maps" and "orca_no_maps" share the same metadata procedure
        assert env_name in {"orca_maps", "orca_no_maps"}, "Unknown env for ORCA!"

        dataset_parts: List[Tuple[str, ...]] = [
                ("train", "val", "test",),
        ]

        # Using seeded randomness to assign 80% of scenes to "train" and 10% to "val" and "test"
        rng = Random(0)
        ntrain = int(0.8*ORCA_NUM_SCENES)
        nval = int(0.1*ORCA_NUM_SCENES)
        ntest = ORCA_NUM_SCENES - (ntrain + nval)
        scene_split = ["train"] * ntrain + ["val"] * nval + ["test"] * ntest
        rng.shuffle(scene_split)

        # both environments share scene names, so need to uniquely identify somehow
        self.all_scene_names = [scene_name + '_%s' % (env_name) for scene_name in ALL_SCENES]

        scene_split_map = {
            k : scene_split[idx] for idx, k in enumerate(self.all_scene_names)
        }

        return EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=ORCA_DT,
            parts=dataset_parts,
            scene_split_map=scene_split_map,
        )

    def load_dataset_obj(self, verbose: bool = False) -> None:
        if verbose:
            print(f"Loading {self.name} dataset...", flush=True)

        scenes_pref = "maps" if self.name == "orca_maps" else "no_maps"

        # full dataset fits in memory so can load everything into dataset obj
        agent_data: Dict[str, Dict[str, np.array]] = OrderedDict()
        map_data: Dict[str, List[List[Tuple[float]]]] = OrderedDict()
        for scene_name, data_name in zip(self.all_scene_names, ALL_SCENES):
            scene_dir: Path = Path(self.metadata.data_dir) / scenes_pref / data_name
            # always load in sim data
            sim_path: Path = scene_dir / "sim.npz"
            sim_data = np.load(sim_path)
            sim_dict: Dict[str, np.array] = {
                "ts" : sim_data["trackTime"],
                "pos" : sim_data["trackPos"],
                "vel" : sim_data["trackVel"],
                "radius" : sim_data["radius"].reshape((-1,)),
            }
            agent_data[scene_name] = sim_dict
            # load in map if available
            if self.name == "orca_maps":
                map_path: Path = scene_dir / "map.pkl"
                with open(map_path, "rb") as mapf:
                    map_obj = pickle.load(mapf)
                map_data[scene_name] = map_obj

        self.dataset_obj = {
            "agents" : agent_data
        }
        if self.name == "orca_maps":
            self.dataset_obj["maps"] = map_data

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        # ALL scenes in the dataset (needs to be cached)
        all_scenes_list: List[OrcaRecord] = list()
        # ONLY RELEVANT scenes that match the tag (is returned)
        scenes_list: List[SceneMetadata] = list()
        for idx, (scene_name, sim_dict) in enumerate(self.dataset_obj["agents"].items()):
            # we have a different map/location for EVERY scene
            # so the location is not included in the scene tags/records, just use the name
            scene_split: str = self.metadata.scene_split_map[scene_name]
            scene_length: int = sim_dict["ts"].shape[0]
            
            # Saving all scene records for later caching.
            all_scenes_list.append(
                OrcaRecord(scene_name, scene_length, idx)
            )

            if (
                scene_split in scene_tag
                and scene_desc_contains is None
            ):
                scene_metadata = SceneMetadata(
                    env_name=self.metadata.name,
                    name=scene_name,
                    dt=self.metadata.dt,
                    raw_data_idx=idx,
                )
                scenes_list.append(scene_metadata)

        self.cache_all_scenes_list(env_cache, all_scenes_list)
        return scenes_list

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[Scene]:
        all_scenes_list: List[OrcaRecord] = env_cache.load_env_scenes_list(self.name)

        scenes_list: List[Scene] = list()
        for scene_record in all_scenes_list:
            (
                scene_name,
                scene_length,
                data_idx,
            ) = scene_record
            scene_split: str = self.metadata.scene_split_map[scene_name]

            if (
                scene_split in scene_tag
                and scene_desc_contains is None
            ):
                scene_metadata = Scene(
                    self.metadata,
                    scene_name,
                    scene_name, # location is just the name since every scene is different
                    scene_split,
                    scene_length,
                    data_idx,
                    None,  # This isn't used if everything is already cached.
                )
                scenes_list.append(scene_metadata)

        return scenes_list

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        _, scene_name, _, data_idx = scene_info

        scene_split: str = self.metadata.scene_split_map[scene_name]
        scene_length: int = self.dataset_obj["agents"][scene_name]["ts"].shape[0]

        return Scene(
            self.metadata,
            scene_name,
            scene_name, # location is just the name since every scene is different
            scene_split,
            scene_length,
            data_idx,
            None,  # No data access info necessary
        )

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        # all agents are visible in every frame, so processing is straightforward
        scene_raw_data = self.dataset_obj["agents"][scene.name].copy()

        #
        # First cache data for the scene
        #
        scene_pos = scene_raw_data["pos"]
        scene_vel = scene_raw_data["vel"]
        # Doing this prepending so that the first acceleration isn't zero
        #       (rather it's just the first actual acceleration duplicated)
        prepend_vel = scene_vel[:,0] - (scene_vel[:,1] - scene_vel[:,0])
        scene_accel = (
            np.diff(scene_vel, axis=1, prepend=prepend_vel[:,np.newaxis,:])
            / ORCA_DT
        )
        # This is likely to be inaccurate/noisy...
        scene_yaw = np.arctan2(scene_vel[:,:,1:2], scene_vel[:,:,0:1])

        scene_data_np = np.concatenate([scene_pos, scene_vel, scene_accel, scene_yaw], axis=-1)
        num_agents, num_t, _ = scene_data_np.shape
        scene_data = pd.DataFrame(
            scene_data_np.reshape((num_agents*num_t, -1)),
            columns=["x", "y", "vx", "vy", "ax", "ay", "heading"],
            index=pd.MultiIndex.from_product(
                [np.arange(num_agents).astype(str).tolist(), range(num_t)],
                names=["agent_id", "scene_ts"],
            ),
        )

        cache_class.save_agent_data(
            scene_data,
            cache_path,
            scene,
        )

        #
        # Then compute per-agent metadata
        #
        agent_list: List[AgentMetadata] = list()
        agent_presence: List[List[AgentMetadata]] = [
            [] for _ in range(num_t)
        ]
        for agent_id, agent_rad in enumerate(scene_raw_data["radius"]):
            agent_diam = agent_rad*2
            agent_metadata = AgentMetadata(
                    name=str(agent_id),
                    agent_type=AgentType.PEDESTRIAN,
                    first_timestep=0,
                    last_timestep=num_t-1,
                    extent=FixedExtent(agent_diam, agent_diam, 1.75),
                )
            agent_list.append(agent_metadata)
            for frame in range(num_t):
                agent_presence[frame].append(agent_metadata)

        return agent_list, agent_presence

    def cache_map(
        self,
        map_name: str,
        layer_names: List[str],
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        resolution: float,
    ) -> None:
        """
        layer_names should contain be ["walkable", "obstacles"]
        resolution is in pixels per meter.
        """
        if "maps" not in self.dataset_obj or map_name not in self.dataset_obj["maps"]:
            return
        if map_cache_class.is_map_cached(cache_path, self.name, map_name, resolution):
            return

        orca_map = self.dataset_obj["maps"][map_name]

        height_m, width_m  = ORCA_MAP_DIM

        height_px, width_px = round(height_m * resolution), round(width_m * resolution)
        # Transformation from world coordinates [m] to map coordinates [px]
        map_from_world: np.ndarray = np.array(
            [[resolution, 0.0, resolution * (width_m / 2.0)],
             [0.0, resolution, resolution * (height_m / 2.0)],
             [0.0, 0.0, 1.0]]
        )

        def layer_fn(layer_name: str) -> np.ndarray:
            return get_map_mask(
                orca_map,
                layer_name,
                map_from_world,
                [height_px, width_px]
            ).astype(np.bool)

        map_shape = (len(layer_names), height_px, width_px)
        map_info: MapMetadata = MapMetadata(
            name=map_name,
            shape=map_shape,
            layers=layer_names,
            layer_rgb_groups=([1], [0], [1]),
            resolution=resolution,
            map_from_world=map_from_world,
        )
        map_cache_class.cache_map_layers(cache_path, map_info, layer_fn, self.name)

    def cache_maps(
        self, cache_path: Path, map_cache_class: Type[SceneCache], resolution: float
    ) -> None:
        """
        Stores rasterized maps to disk for later retrieval.
        """
        layer_names: List[str] = [
            "walkable",
            "obstacles"
        ]
        # can only cache if "orca_maps" dataset
        if "maps" in self.dataset_obj:
            for map_name in tqdm(
                self.dataset_obj["maps"].keys() , desc=f"Caching {self.name} Maps at {resolution} px/m"
            ):
                self.cache_map(
                    map_name, layer_names, cache_path, map_cache_class, resolution
                )

def get_map_mask(map_data, layer_name, map_from_world, canvas_size):
    '''
    Rasterizes an ORCA sim map.
    - obstacles: ORCA map data (list of obstacle polygons)
    - layer_name: one of ["walkable", "obstacles"] supported.
    - map_from_world: 3x3 transformation matrix from world coordinates [m] to map coordinates [px]
    - canvas_size tuple: (H, W) pixels tuple which determines the rasterized image size
    '''
    init_val = 1 if layer_name == "walkable" else 0
    map_mask = np.ones(canvas_size, np.uint8) * init_val
    for obstacles in map_data:
        cur_obs = np.array(obstacles)
        # convert to pixel coords
        poly_pts = np.concatenate([cur_obs, np.ones((cur_obs.shape[0], 1))], axis=1)
        poly_pts = map_from_world @ poly_pts.T
        poly_pts = poly_pts[:-1, :].T
        # draw polygon
        coords = poly_pts.round().astype(np.int32)
        cv2.fillPoly(map_mask, [coords], 0 if layer_name == "walkable" else 1)

    return map_mask