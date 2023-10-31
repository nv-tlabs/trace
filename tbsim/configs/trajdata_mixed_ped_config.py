import math
import numpy as np

from tbsim.configs.trajdata_config import TrajdataTrainConfig, TrajdataEnvConfig


class MixedPedTrainConfig(TrajdataTrainConfig):
    def __init__(self):
        super(MixedPedTrainConfig, self).__init__()

        self.trajdata_cache_location = "~/.unified_data_cache"
        self.trajdata_source_train = ["nusc_trainval-train", "eupeds_eth-train", "eupeds_hotel-train", "eupeds_univ-train", "eupeds_zara1-train", "eupeds_zara2-train"]
        self.trajdata_source_valid = ["nusc_trainval-train_val", "eupeds_eth-val", "eupeds_hotel-val", "eupeds_univ-val", "eupeds_zara1-val", "eupeds_zara2-val"]
        # dict mapping dataset IDs -> root path
        #       all datasets that will be used must be included here
        self.trajdata_data_dirs = {
            "nusc_trainval" : "./datasets/nuscenes",
            "nusc_test" : "./datasets/nuscenes",
            "nusc_mini" : "./datasets/nuscenes",
            "eupeds_eth" : "./datasets/eth_ucy", 
            "eupeds_hotel" : "./datasets/eth_ucy",
            "eupeds_univ" : "./datasets/eth_ucy",
            "eupeds_zara1" : "./datasets/eth_ucy",
            "eupeds_zara2" : "./datasets/eth_ucy"
        }

        # for debug
        self.trajdata_rebuild_cache = False

        # training config
        self.training.batch_size = 200
        self.training.num_steps = 100000
        self.training.num_data_workers = 6

        self.save.every_n_steps = 3000
        self.save.best_k = 5

        # validation config
        self.validation.enabled = True
        self.validation.batch_size = 32
        self.validation.num_data_workers = 4
        self.validation.every_n_steps = 1444
        self.validation.num_steps_per_epoch = 500

        self.logging.terminal_output_to_txt = True  # whether to log stdout to txt file
        self.logging.log_tb = False  # enable tensorboard logging
        self.logging.log_wandb = True  # enable wandb logging
        self.logging.wandb_project_name = "tbsim"
        self.logging.log_every_n_steps = 10
        self.logging.flush_every_n_steps = 100


class MixedPedEnvConfig(TrajdataEnvConfig):
    def __init__(self):
        super(MixedPedEnvConfig, self).__init__()

        #
        # with map, non-rasterized history
        #
        self.data_generation_params.trajdata_incl_map = True
        self.data_generation_params.trajdata_max_agents_distance = 15.0
        self.rasterizer.num_sem_layers = 7
        self.rasterizer.drivable_layers = [] #[0, 1, 2] every layer is "drivable" for a pedestrian
        self.rasterizer.include_hist = False 

        # which types of neighbor agents
        # self.data_generation_params.trajdata_only_types = ["vehicle", "pedestrian", "bicycle", "motorcycle"]
        self.data_generation_params.trajdata_only_types = ["pedestrian"]
        # which types of agents to predict
        self.data_generation_params.trajdata_predict_types = ["pedestrian"]

        # NOTE: rasterization info must still be provided even if incl_map=False
        #       since still used for agent states
        # how to group layers together to viz RGB image
        self.rasterizer.rgb_idx_groups = ([0, 1, 2], [3, 4], [5, 6])
        # raster image size [pixels]
        self.rasterizer.raster_size = 224
        # raster's spatial resolution [meters per pixel]: the size in the real world one pixel corresponds to.
        self.rasterizer.pixel_size = 1.0 / 12.0 # 12 px/m
        # where the agent is on the map, (0.0, 0.0) is the center
        self.rasterizer.ego_center = (-0.5, 0.0)
        # if incl_map = True, but no map is available, will fill dummy map with this value
        self.rasterizer.no_map_fill_value = 0.5 # -1.0