import random

import numpy as np
from copy import deepcopy

import numpy as np

from tbsim.configs.config import Dict
from tbsim.configs.eval_config import EvaluationConfig

class SceneEditingConfig(EvaluationConfig):
    def __init__(self):
        super(SceneEditingConfig, self).__init__()
        #
        # The most relevant args from EvaluationConfig. For rest, see that file.
        #
        self.name = "scene_edit_eval"
        self.eval_class = "Diffuser" 
        self.env = "trajdata" # only supported environment right now
        self.results_dir = "scene_edit_eval/"
        self.num_scenes_per_batch = 1

        # number of trajectories samples from the diffusion model
        self.policy.num_action_samples = 10

        # if True, computes guidance loss only after full denoising and only uses
        #       to choose the action, not to get gradient to guide
        self.policy.guide_as_filter_only = False
        # if True, chooses the sample that's closest to GT at each planning step
        self.policy.guide_with_gt = False

        #
        # diffuser-only options
        #
        # if > 0.0 uses classifier-free guidance (mix of conditional and non-cond)
        # model at test time. Uses drop_fill value above.
        self.policy.class_free_guide_w = 0.0
        # whether to guide the predicted CLEAN or NOISY trajectory at each step
        self.policy.guide_clean = True # uses clean ("reconstruction") guidance if true

        self.metrics.compute_analytical_metrics = True

        self.trajdata.trajdata_cache_location = "~/.unified_data_cache"
        self.trajdata.trajdata_rebuild_cache = False
        # number of simulations to run in each scene
        #       if > 1, each sim is running from a different starting point in the scene
        self.trajdata.num_sim_per_scene = 1

        #
        # NOTE: by default, the config for ORCA data is uncommented
        #       comment this out and uncomment below for using ETH/UCY (eupeds) and nuscenes datasets instead

        #
        ## orca
        #
        self.trajdata.trajdata_source_test = ["orca_maps-test"]
        self.trajdata.trajdata_data_dirs = {
            "orca_maps" : "./datasets/orca_sim",
            "orca_no_maps" : "./datasets/orca_sim",
        }
        self.trajdata.num_scenes_to_evaluate = 100
        self.trajdata.eval_scenes = np.arange(100).tolist()
        self.trajdata.n_step_action = 50
        self.trajdata.num_simulation_steps = 50
        self.trajdata.skip_first_n = 0

        
        # ## eupeds
        # #
        # # self.trajdata.trajdata_source_test = ["eupeds_eth-test_loo"]
        # self.trajdata.trajdata_source_test = ["eupeds_eth-val", 
        #                                       "eupeds_hotel-val",
        #                                       "eupeds_univ-val",
        #                                       "eupeds_zara1-val",
        #                                       "eupeds_zara2-val"]
        # self.trajdata.trajdata_data_dirs = {
        #     "eupeds_eth" : "./datasets/eth_ucy", 
        #     "eupeds_hotel" : "./datasets/eth_ucy",
        #     "eupeds_univ" : "./datasets/eth_ucy",
        #     "eupeds_zara1" : "./datasets/eth_ucy",
        #     "eupeds_zara2" : "./datasets/eth_ucy"
        # }
        # self.trajdata.num_scenes_to_evaluate = 6
        # self.trajdata.eval_scenes = np.arange(6).tolist()
        # self.trajdata.n_step_action = 50
        # self.trajdata.num_simulation_steps = 50
        # self.trajdata.skip_first_n = 0
        # self.trajdata.num_sim_per_scene = 20

        
        # ## nusc
        # #
        # self.trajdata.trajdata_source_test = ["nusc_trainval-val"]
        # self.trajdata.trajdata_data_dirs = {
        #     "nusc_trainval" : "./datasets/nuscenes",
        # }
        # # 118 val scenes contain pedestrians
        # self.trajdata.eval_scenes = np.arange(118).tolist()
        # self.trajdata.num_scenes_to_evaluate = len(self.trajdata.eval_scenes)
        # self.trajdata.n_step_action = 10
        # self.trajdata.num_simulation_steps = 100
        # self.trajdata.skip_first_n = 0


        self.edits.editing_source = ['heuristic'] # [config, heuristic, None]
        # self.edits.editing_source = [None] # [config, heuristic, None]
        self.edits.guidance_config = []

        # 
        # NOTE: Just an example for ORCA data, see configs for other ways to set this
        #
        self.edits.heuristic_config = [
            {
             'name' : 'agent_collision',
             'weight' : 1000.0,
             'params' : {
                            'num_disks' : 1,        # to approximate agents
                            'buffer_dist' : 0.2,     # extra social distance
                        }
            },
            {
             'name' : 'map_collision',
             'weight' : 10.0,
             'params' : {
                            'num_points_lw' : (10, 10),
                        }
            },
            {
             'name' : 'target_pos_at_time',
             'weight' : 30000.0,
             'params' : {
                            'target_time' : 40
                        },
            }
        ]

    def clone(self):
        return deepcopy(self)
