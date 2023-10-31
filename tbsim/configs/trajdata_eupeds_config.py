import math
import numpy as np

from tbsim.configs.base import TrainConfig, EnvConfig, AlgoConfig
from tbsim.configs.trajdata_config import TrajdataTrainConfig, TrajdataEnvConfig


class EupedsTrainConfig(TrajdataTrainConfig):
    def __init__(self):
        super(EupedsTrainConfig, self).__init__()

        self.trajdata_cache_location = "~/.unified_data_cache"
        # leaves out the ETH-Univ dataset for training
        self.trajdata_source_train = ["eupeds_eth-train_loo"]
        self.trajdata_source_valid = ["eupeds_eth-val_loo"]
        # dict mapping dataset IDs -> root path
        #       all datasets that will be used must be included here
        self.trajdata_data_dirs = {
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
        self.training.num_data_workers = 4

        self.save.every_n_steps = 3000
        self.save.best_k = 5

        # validation config
        self.validation.enabled = True
        self.validation.batch_size = 32
        self.validation.num_data_workers = 4
        self.validation.every_n_steps = 550
        self.validation.num_steps_per_epoch = 200

        self.logging.terminal_output_to_txt = True  # whether to log stdout to txt file
        self.logging.log_tb = False  # enable tensorboard logging
        self.logging.log_wandb = True  # enable wandb logging
        self.logging.wandb_project_name = "tbsim"
        self.logging.log_every_n_steps = 10
        self.logging.flush_every_n_steps = 100


class EupedsEnvConfig(TrajdataEnvConfig):
    def __init__(self):
        super(EupedsEnvConfig, self).__init__()

        #
        # # no map, non-rasterized history
        # #
        # self.data_generation_params.trajdata_incl_map = False
        # self.data_generation_params.trajdata_max_agents_distance = 15.0
        # self.rasterizer.num_sem_layers = 0
        # self.rasterizer.include_hist = False

        #
        # no map, non-rasterized history
        #
        self.data_generation_params.trajdata_incl_map = True
        self.data_generation_params.trajdata_max_agents_distance = 15.0
        self.rasterizer.num_sem_layers = 7
        self.rasterizer.include_hist = False # depends on the model being used
        self.rasterizer.drivable_layers = [] 
        self.rasterizer.rgb_idx_groups = ([0, 1, 2], [3, 4], [5, 6])

        # no maps to include
        self.data_generation_params.trajdata_only_types = ["pedestrian"]

        # NOTE: rasterization info must still be provided even if incl_map=False
        #       since still used for agent states
        # raster image size [pixels]
        self.rasterizer.raster_size = 224
        # raster's spatial resolution [meters per pixel]: the size in the real world one pixel corresponds to.
        self.rasterizer.pixel_size = 1. / 12.
        # where the agent is on the map, (0.0, 0.0) is the center and image width is 2.0, i.e. (1.0, 0.0) is the right edge
        self.rasterizer.ego_center = (-0.5, 0.0)
        # if incl_map = True, but no map is available, will fill dummy map with this value
        self.rasterizer.no_map_fill_value = 0.5 # -1.0