import numpy as np
from copy import deepcopy

from tbsim.configs.config import Dict


class EvaluationConfig(Dict):
    def __init__(self):
        super(EvaluationConfig, self).__init__()
        self.name = None
        self.env = "trajdata"
        self.dataset_path = None
        self.eval_class = ""
        self.seed = 0
        self.num_scenes_per_batch = 1
        self.num_scenes_to_evaluate = 1

        self.num_episode_repeats = 1
        self.start_frame_index_each_episode = None  # if specified, should be the same length as num_episode_repeats
        self.seed_each_episode = None  # if specified, should be the same length as num_episode_repeats

        self.ego_only = False # needed for training rollout callback

        self.ckpt_root_dir = "checkpoints/"
        self.experience_hdf5_path = None
        self.results_dir = "results/"

        self.ckpt.policy.ckpt_dir = None
        self.ckpt.policy.ckpt_key = None

        self.policy.num_action_samples = 10

        self.metrics.compute_analytical_metrics = True

        self.trajdata.trajdata_cache_location = "~/.unified_data_cache"
        self.trajdata.trajdata_rebuild_cache = False

        #
        # eupeds
        # 

        # self.trajdata.trajdata_source_test = ["eupeds_eth-val", "eupeds_hotel-val", "eupeds_univ-val", "eupeds_zara1-val", "eupeds_zara2-val"]
        # self.trajdata.trajdata_data_dirs = {
        #     "eupeds_eth" : "./datasets/eth_ucy", 
        #     "eupeds_hotel" : "./datasets/eth_ucy",
        #     "eupeds_univ" : "./datasets/eth_ucy",
        #     "eupeds_zara1" : "./datasets/eth_ucy",
        #     "eupeds_zara2" : "./datasets/eth_ucy"
        # }
        # self.trajdata.num_scenes_to_evaluate = 6
        # self.trajdata.eval_scenes = np.arange(6).tolist()
        # self.trajdata.n_step_action = 2
        # self.trajdata.num_simulation_steps = 25
        # self.trajdata.skip_first_n = 0

        #
        # orca
        #

        self.trajdata.trajdata_source_test = ["orca_maps-test"]
        self.trajdata.trajdata_data_dirs = {
            "orca_maps" : "./datasets/orca_sim",
            "orca_no_maps" : "./datasets/orca_sim",
        }
        self.trajdata.num_scenes_to_evaluate = 200
        self.trajdata.eval_scenes = np.arange(200).tolist()
        self.trajdata.n_step_action = 1 #5
        self.trajdata.num_simulation_steps = 100
        self.trajdata.skip_first_n = 0
        self.policy.num_action_samples = 10
        
        #
        # nusc
        #

        # self.trajdata.trajdata_source_test = ["nusc_trainval-val"]
        # self.trajdata.trajdata_data_dirs = {
        #     "nusc_trainval" : "./datasets/nuscenes",
        # }
        # self.trajdata.num_scenes_to_evaluate = 100
        # self.trajdata.eval_scenes = np.arange(100).tolist()
        # self.trajdata.n_step_action = 5
        # self.trajdata.num_simulation_steps = 100
        # self.trajdata.skip_first_n = 0

    def clone(self):
        return deepcopy(self)
