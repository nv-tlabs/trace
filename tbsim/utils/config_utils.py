import json
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.configs.base import ExperimentConfig
from tbsim.configs.config import Dict


def get_experiment_config_from_file(file_path, locked=False):
    ext_cfg = json.load(open(file_path, "r"))
    cfg = get_registered_experiment_config(ext_cfg["registered_name"])
    cfg.update(**ext_cfg)
    cfg.lock(locked)
    return cfg

def translate_pass_trajdata_cfg(cfg: ExperimentConfig):
    """
    Translate a unified passthrough config to trajdata.
    """
    rcfg = Dict()
    rcfg.step_time = cfg.algo.step_time
    rcfg.trajdata_cache_location = cfg.train.trajdata_cache_location
    rcfg.trajdata_source_train = cfg.train.trajdata_source_train
    rcfg.trajdata_source_valid = cfg.train.trajdata_source_valid
    rcfg.trajdata_data_dirs = cfg.train.trajdata_data_dirs
    rcfg.trajdata_rebuild_cache = cfg.train.trajdata_rebuild_cache

    rcfg.history_num_frames = cfg.algo.history_num_frames
    rcfg.future_num_frames = cfg.algo.future_num_frames

    rcfg.trajdata_centric = cfg.env.data_generation_params.trajdata_centric
    rcfg.trajdata_only_types = cfg.env.data_generation_params.trajdata_only_types
    rcfg.trajdata_predict_types = cfg.env.data_generation_params.trajdata_predict_types
    rcfg.trajdata_incl_map = cfg.env.data_generation_params.trajdata_incl_map
    rcfg.max_agents_distance = cfg.env.data_generation_params.trajdata_max_agents_distance
    rcfg.trajdata_standardize_data = cfg.env.data_generation_params.trajdata_standardize_data
    rcfg.trajdata_scene_desc_contains = cfg.env.data_generation_params.trajdata_scene_desc_contains

    rcfg.pixel_size = cfg.env.rasterizer.pixel_size
    rcfg.raster_size = int(cfg.env.rasterizer.raster_size)
    rcfg.raster_center = cfg.env.rasterizer.ego_center
    rcfg.num_sem_layers = cfg.env.rasterizer.num_sem_layers
    rcfg.drivable_layers = cfg.env.rasterizer.drivable_layers
    rcfg.no_map_fill_value = cfg.env.rasterizer.no_map_fill_value
    rcfg.raster_include_hist = cfg.env.rasterizer.include_hist

    rcfg.lock()
    return rcfg
