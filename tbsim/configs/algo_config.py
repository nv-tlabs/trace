import math

from tbsim.configs.base import AlgoConfig

class DiffuserConfig(AlgoConfig):
    def __init__(self):
        super(DiffuserConfig, self).__init__()
        self.eval_class = "Diffuser"
        self.name = "diffuser"
        
        ## model
        self.map_encoder_model_arch = "resnet18"
        self.diffuser_model_arch = "TemporalMapUnet"

        # whether the map will be passed in as conditioning
        #       i.e. do we need a map encoder
        self.rasterized_map = True
        # whether to use a "global" map feature (passed in as conditioning to diffuser)
        #   and/or a feature grid (sampled at each trajectory position and concated to trajectory input to diffuser)
        self.use_map_feat_global = False
        self.use_map_feat_grid = True

        self.base_dim = 32 # time embedding size and hidden size (multiplied by dim_mults)
        self.horizon = 52 # how many steps in future to predict
        self.n_diffusion_steps = 100
        self.action_weight = 1 #10 in OG diffuser
        self.loss_discount = 1 # apply same loss over whole trajectory, don't lessen loss further in future
        self.dim_mults = (2, 4, 8) # (1, 4, 8) # defines the channel size at layers of diffuser convs
        self.loss_type = 'l2'

        #
        # Exponential moving average
        self.use_ema = True
        self.ema_step = 1 #10
        self.ema_decay = 0.995
        self.ema_start_step = 4000 # 2000 -- smaller batch size for real-world data        

        # ['state', 'action', 'state_and_action', 'state_and_action_no_dyn']
        self.diffuser_input_mode = 'state_and_action'

        # likelihood of not using conditioning as input, even if available
        #       if 1.0, doesn't include cond encoder in arch
        #       if 0.0, conditioning is always used as usual
        self.conditioning_drop_map_p = 0.1
        self.conditioning_drop_neighbor_p = 0.1
        # value to fill in when condition is "dropped". Should not just be 0 -- the model
        #   should "know" data is missing, not just think there are e.g. no obstacles in the map.
        #   NOTE: this should be the same as the value given to trajdata to fill in missing map data
        self.conditioning_drop_fill = 0.5 # -1, 1, and 0 all show up in map or neighbor history

        # the final conditioning feature size after all cond inputs
        #       have been processed together (e.g. map feat, hist feat, state feat...)
        self.cond_feat_dim = 256
        self.map_feature_dim = 256
        self.map_grid_feature_dim = 32

        self.history_feature_dim = 128 # if separate from map
        self.history_num_frames = 30
        self.history_num_frames_ego = 30
        self.history_num_frames_agents = 30
        self.future_num_frames = self.horizon
        self.step_time = 0.1

        self.dynamics.type = "Unicycle"
        self.dynamics.max_steer = 0.5
        self.dynamics.max_yawvel = math.pi * 2.0
        self.dynamics.acce_bound = (-10, 8)
        self.dynamics.ddh_bound = (-math.pi * 2.0, math.pi * 2.0)
        self.dynamics.max_speed = 40.0  # roughly 90mph

        self.loss_weights.diffusion_loss = 1.0

        self.optim_params.policy.learning_rate.initial = 2e-4  # policy learning rate

        self.optim_params.policy.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.policy.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs

        # how many samples to take for validation (during training)
        self.diffuser.num_eval_samples = 10

        #
        # how to normalize input data (x, y, vel, yaw, acc, yawvel)
        # NOTE: ORCA is uncommented by default. But if you are generating a new config
        #       or using this config class directly (i.e. not a json file) then make sure
        #       to uncomment for the dataset you're using
        #

        #
        # ORCA
        #
        
        # diffuser denoising model (ego future)
        #                          [    x,        y,       vel,      yaw,     acc,    yawvel ]
        self.diffuser_norm_info = ([-3.538049, 0.004175, -1.360894, 0.001894, 0.015233, 0.000562],
                                   [ 2.304491, 0.462847, 0.426683, 0.191930, 0.255089, 0.175583])
        # agent (ego) history encoding model
        #                            [    x,        y,       vel,      len,     width    ]
        self.agent_hist_norm_info = ([   2.074730, -0.000102, -1.401319,    0.0,       0.0    ],
                                     [   1.376582, 0.214366, 0.369064,    1.0,       1.0    ])
        # neighbor history encoding model
        #                               [    x,        y,       vel,      len,     width    ]
        self.neighbor_hist_norm_info = ([   -2.804952, -0.037075, -0.544995,    0.0,       0.0    ],
                                        [   6.161589,   6.486891,  0.771583,    1.0,       1.0    ])
        
        # #
        # # mixed ETH/UCY + nusc (pedestrian only neighbors)
        # #

        # # diffuser denoising model (ego future)
        # #                          [    x,        y,       vel,      yaw,     acc,    yawvel ]
        # self.diffuser_norm_info = ([-1.985679, -0.002455, -0.777417, 0.002131, 0.002807, 0.000199],
        #                            [ 2.111840, 0.415881, 0.617595, 0.342887, 0.387015, 0.230898 ])
        # # agent (ego) history encoding model
        # #                            [    x,        y,       vel,      len,     width    ]
        # self.agent_hist_norm_info = ([1.146453, 0.003049, -0.798847, -0.662612, -0.614196],
        #                              [1.242787, 0.194718, 0.601530, 0.193939, 0.136501])
        # # neighbor history encoding model
        # #                               [    x,        y,       vel,      len,     width    ]
        # self.neighbor_hist_norm_info = ([  -0.282255, -0.034856, -0.638080, -0.526286, -0.521026 ],
        #                                 [  5.180557, 4.354133, 0.473684, 0.081178, 0.063327  ])

        # #
        # # ETH/UCY
        # #

        # # diffuser denoising model (ego future)
        # #                          [    x,        y,       vel,      yaw,     acc,    yawvel ]
        # self.diffuser_norm_info = ([-1.279591, -0.003254, -0.531359, 0.005799, -0.005604, 0.000634],
        #                            [1.428520, 0.464483, 0.436589, 0.572977, 0.253401, 0.397780])
        # # agent (ego) history encoding model
        # #                            [    x,        y,       vel,      len,     width    ]
        # self.agent_hist_norm_info = ([   0.772782, 0.002752, -0.546067,    0.0,       0.0    ],
        #                              [ 0.859441, 0.162332, 0.442286,    1.0,       1.0    ])
        # # neighbor history encoding model
        # #                               [    x,        y,       vel,      len,     width    ]
        # self.neighbor_hist_norm_info = ([   -0.360387, -0.030692, -0.317841,   0.0,       0.0    ],
        #                                 [  4.098931, 3.562263, 0.435449,    1.0,       1.0    ])

        
        # #
        # # nusc only (pedestrian only neighbors)
        # #

        # # diffuser denoising model (ego future)
        # #                          [    x,        y,       vel,      yaw,     acc,    yawvel ]
        # self.diffuser_norm_info = ([-2.368314, -0.001920, -0.914953, 0.001183, 0.005788, -0.000092],
        #                            [2.282809, 0.380954, 0.657391, 0.204417, 0.464515, 0.104958 ])
        # # agent (ego) history encoding model
        # #                            [    x,        y,       vel,      len,     width    ]
        # self.agent_hist_norm_info = ([1.354268, 0.003156, -0.941192, -0.756262, -0.678425],
        #                              [1.377433, 0.211902, 0.640642, 0.190733, 0.133160])
        # # neighbor history encoding model
        # #                               [    x,        y,       vel,      len,     width    ]
        # self.neighbor_hist_norm_info = ([   0.076816, -0.053561, -0.760115, -0.735165, -0.732473 ],
        #                                 [   6.480056, 4.817492, 0.635160, 0.178575, 0.147555 ])

        # #
        # # nusc only (all agent type neighbors)
        # #
        # # diffuser denoising model (ego future)
        # #                          [    x,        y,       vel,      yaw,     acc,    yawvel ]
        # self.diffuser_norm_info = ([-2.368314, -0.001920, -0.914953, 0.001183, 0.005788, -0.000092],
        #                            [2.282809, 0.380954, 0.657391, 0.204417, 0.464515, 0.104958 ])
        # # agent (ego) history encoding model
        # #                            [    x,        y,       vel,      len,     width    ]
        # self.agent_hist_norm_info = ([1.354268, 0.003156, -0.941192, -0.756262, -0.678425],
        #                              [1.377433, 0.211902, 0.640642, 0.190733, 0.133160])
        # # neighbor history encoding model
        # #                               [    x,        y,       vel,      len,     width    ]
        # self.neighbor_hist_norm_info = ([   -0.062956, -0.084551, -0.945207, -1.484518, -0.831468 ],
        #                                 [   7.026883, 5.912263, 1.144973, 1.735821, 0.631578 ])