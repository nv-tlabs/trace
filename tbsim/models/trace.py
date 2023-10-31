#
# Implementation based on Diffuser: https://github.com/jannerm/diffuser/blob/main/diffuser/models/diffusion.py
#   which is based on denoising-diffusion-pytorch: https://github.com/lucidrains/denoising-diffusion-pytorch
#

from typing import Dict, List
from collections import OrderedDict
import torch
import torch.nn as nn

import tbsim.utils.tensor_utils as TensorUtils
import tbsim.models.base_models as base_models
from tbsim.utils.batch_utils import batch_utils
from tbsim.utils.geometry_utils import transform_points_tensor

from tbsim.utils.guidance_loss import DiffuserGuidance, verify_guidance_config_list

import numpy as np
from .trace_helpers import (
    cosine_beta_schedule,
    extract,
    Losses,
    convert_state_to_state_and_action,
    query_feature_grid,
    unicyle_forward_dynamics,
    AgentHistoryEncoder,
    NeighborHistoryEncoder,
    MapEncoder,
)
from .temporal import TemporalMapUnet
import tbsim.dynamics as dynamics

class DiffuserModel(nn.Module):
    '''
    TRACE model.
    '''
    def __init__(
            self,
            map_encoder_model_arch: str,
            input_image_shape,
            map_feature_dim: int,
            map_grid_feature_dim: int,
            diffuser_model_arch: str,
            horizon: int,
            observation_dim: int, 
            action_dim: int,
            output_dim: int,
            cond_feature_dim = 256,
            rasterized_map = True,
            use_map_feat_global = False,
            use_map_feat_grid = True,
            hist_num_frames = 31,
            hist_feature_dim = 128,
            n_timesteps=1000,
            loss_type='l2', 
            action_weight=1.0, 
            loss_discount=1.0, 
            dim_mults=(1, 2, 4, 8),
            dynamics_type=None,
            dynamics_kwargs={},
            base_dim=32,
            diffuser_input_mode='state_and_action',
            use_conditioning=True,
            cond_fill_value=-1.0,
            # norm info is ([add_coeffs, div_coeffs])
            diffuser_norm_info=([-17.5, 0, 0, 0, 0, 0],[22.5, 10, 40, 3.14, 500, 31.4]),
            # if using non-rasterized histories, need these
            agent_hist_norm_info=([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]),
            neighbor_hist_norm_info=([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]),
            dt=0.1,
    ) -> None:

        super().__init__()

        # this applies to map and past NEIGHBOR conditioning only
        #       curr state or past ego trajecotry are always given
        self.use_conditioning = use_conditioning
        # for test-time classifier-free guidance, if desired
        self.cond_fill_value = cond_fill_value 

        self.rasterized_map = rasterized_map

        cond_in_feat_size = 0
        cond_out_feat_size = cond_feature_dim

        # history encoding
        self.agent_hist_encoder = self.neighbor_hist_encoder = None
        # ego history is ALWAYS used as conditioning
        self.agent_hist_encoder = AgentHistoryEncoder(hist_num_frames,
                                                        out_dim=hist_feature_dim,
                                                        use_norm=True,
                                                        norm_info=agent_hist_norm_info)
        cond_in_feat_size += hist_feature_dim

        if self.use_conditioning:
            self.neighbor_hist_encoder = NeighborHistoryEncoder(hist_num_frames,
                                                                out_dim=hist_feature_dim,
                                                                use_norm=True,
                                                                norm_info=neighbor_hist_norm_info)
            cond_in_feat_size += hist_feature_dim

        # map encoding
        self.map_encoder = None
        self.use_map_feat_global = use_map_feat_global
        self.use_map_feat_grid = use_map_feat_grid
        self.input_image_shape = input_image_shape
        if self.use_conditioning and self.rasterized_map:
            self.map_encoder = MapEncoder(
                model_arch=map_encoder_model_arch,
                input_image_shape=input_image_shape,
                global_feature_dim=map_feature_dim if self.use_map_feat_global else None,
                grid_feature_dim=map_grid_feature_dim if self.use_map_feat_grid else None,
            )

            if self.use_map_feat_global:
                cond_in_feat_size += map_feature_dim

        # MLP to combine conditioning from all sources
        combine_layer_dims = (cond_in_feat_size, cond_in_feat_size, cond_out_feat_size, cond_out_feat_size)
        self.process_cond_mlp = base_models.MLP(cond_in_feat_size,
                                                cond_out_feat_size,
                                                combine_layer_dims,
                                                normalization=True)

        self._dynamics_type = dynamics_type
        self._dynamics_kwargs = dynamics_kwargs
        self._create_dynamics()
        
        # ----- diffuser -----
        self.dt = dt
        # x, y, vel, yaw, acc, yawvel
        assert len(diffuser_norm_info) == 2
        norm_add_coeffs = diffuser_norm_info[0]
        norm_div_coeffs = diffuser_norm_info[1]
        assert len(norm_add_coeffs) == 6
        assert len(norm_div_coeffs) == 6
        self.add_coeffs = np.array(norm_add_coeffs).astype('float32')
        self.div_coeffs = np.array(norm_div_coeffs).astype('float32')
        
        self.diffuser_input_mode = diffuser_input_mode

        if diffuser_input_mode == 'state_and_action':
            self.default_chosen_inds = [0, 1, 2, 3, 4, 5]
        else:
            raise
        
        self.horizon = horizon
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.output_dim = output_dim
        
        if diffuser_model_arch == "TemporalMapUnet":
            transition_in_dim = self.transition_dim
            if self.use_map_feat_grid and self.map_encoder is not None:
                # will be appending map features to each step of trajectory
                transition_in_dim += map_grid_feature_dim
            self.model = TemporalMapUnet(horizon=horizon,
                                      transition_dim=transition_in_dim,
                                      cond_dim=cond_out_feat_size,
                                      output_dim=self.output_dim,
                                      dim=base_dim,
                                      dim_mults=dim_mults,
                                    )
        else:
            print('unknown diffuser_model_arch:', diffuser_model_arch)
            raise

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # calculations for class-free guidance
        self.sqrt_alphas_over_one_minus_alphas_cumprod = torch.sqrt(alphas_cumprod / (1.0 - alphas_cumprod))
        self.sqrt_recip_one_minus_alphas_cumprod = 1.0 / torch.sqrt(1. - alphas_cumprod)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

        # for guided sampling
        self.current_guidance = None

    #------------------------------------------ guidance utils ------------------------------------------#

    def set_guidance(self, guidance_config_list, example_batch=None):
        '''
        Instantiates test-time guidance functions using the list of configs (dicts) passed in.
        '''
        if guidance_config_list is not None:
            if len(guidance_config_list) > 0 and verify_guidance_config_list(guidance_config_list):
                print('Instantiating test-time guidance with configs:')
                print(guidance_config_list)
                self.current_guidance = DiffuserGuidance(guidance_config_list, example_batch)

    def update_guidance(self, **kwargs):
        if self.current_guidance is not None:
            self.current_guidance.update(**kwargs)

    def clear_guidance(self):
        self.current_guidance = None

    #------------------------------------------ utility ------------------------------------------#
    def _create_dynamics(self):
        if self._dynamics_type in ["Unicycle", dynamics.DynType.UNICYCLE]:
            self.dyn = dynamics.Unicycle(
                "dynamics",
                max_steer=self._dynamics_kwargs["max_steer"],
                max_yawvel=self._dynamics_kwargs["max_yawvel"],
                acce_bound=self._dynamics_kwargs["acce_bound"]
            )
        else:
            self.dyn = None

    def get_aux_info(self, data_batch, include_class_free_cond=False):
        N = data_batch["history_positions"].size(0)
        device = data_batch["history_positions"].device

        cond_feat_in = torch.empty((N,0)).to(device)
        non_cond_feat_in = torch.empty((N,0)).to(device)

        #
        # current ego state
        #
        # always need this for rolling out actions
        if self._dynamics_type is not None:
            curr_states = batch_utils().get_current_states(data_batch, dyn_type=self.dyn.type())
        else:
            curr_states = None

        #
        # rasterized map
        #
        map_grid_feat = map_grid_feat_non_cond = raster_from_agent = None
        if self.map_encoder is not None:
            image_batch = data_batch["image"]
            map_global_feat, map_grid_feat = self.map_encoder(image_batch)
            if self.use_map_feat_global:
                cond_feat_in = torch.cat([cond_feat_in, map_global_feat], dim=-1)
            if self.use_map_feat_grid and self.map_encoder is not None:
                raster_from_agent = data_batch["raster_from_agent"]

            if include_class_free_cond:
                image_non_cond = torch.ones_like(image_batch) * self.cond_fill_value
                map_global_feat_non_cond, map_grid_feat_non_cond = self.map_encoder(image_non_cond)
                if self.use_map_feat_global:
                    non_cond_feat_in = torch.cat([non_cond_feat_in, map_global_feat_non_cond], dim=-1)

        #
        # ego history
        #
        if self.agent_hist_encoder is not None:
            agent_hist_feat = self.agent_hist_encoder(data_batch["history_positions"],
                                                      data_batch["history_yaws"],
                                                      data_batch["history_speeds"],
                                                      data_batch["extent"],
                                                      data_batch["history_availabilities"])
            cond_feat_in = torch.cat([cond_feat_in, agent_hist_feat], dim=-1)
            if include_class_free_cond:
                # make all agents zero availability
                non_cond_avail = torch.zeros_like(data_batch["history_speeds"]).to(torch.bool) # BxT
                agent_hist_feat_non_cond = self.agent_hist_encoder(data_batch["history_positions"],
                                                                    data_batch["history_yaws"],
                                                                    data_batch["history_speeds"],
                                                                    data_batch["extent"],
                                                                    non_cond_avail)
                non_cond_feat_in = torch.cat([non_cond_feat_in, agent_hist_feat_non_cond], dim=-1)

        #
        # neighbor history
        #

        # neighbor trajectory encoding
        if self.neighbor_hist_encoder is not None:
            neighbor_hist_feat = self.neighbor_hist_encoder(data_batch["all_other_agents_history_positions"],
                                                            data_batch["all_other_agents_history_yaws"],
                                                            data_batch["all_other_agents_history_speeds"],
                                                            data_batch["all_other_agents_extents"],
                                                            data_batch["all_other_agents_history_availabilities"])
            cond_feat_in = torch.cat([cond_feat_in, neighbor_hist_feat], dim=-1)  
            if include_class_free_cond:
                # make all agents zero availability
                non_cond_neighbor_avail = torch.zeros_like(data_batch["all_other_agents_history_speeds"]).to(torch.bool) # BxNxT
                neighbor_hist_feat_non_cond = self.neighbor_hist_encoder(data_batch["all_other_agents_history_positions"],
                                                                        data_batch["all_other_agents_history_yaws"],
                                                                        data_batch["all_other_agents_history_speeds"],
                                                                        data_batch["all_other_agents_extents"],
                                                                        non_cond_neighbor_avail)
                non_cond_feat_in = torch.cat([non_cond_feat_in, neighbor_hist_feat_non_cond], dim=-1)

        #
        # Process all features together
        #
        cond_feat = self.process_cond_mlp(cond_feat_in)
        non_cond_feat = None
        if include_class_free_cond:
            non_cond_feat = self.process_cond_mlp(non_cond_feat_in)

        aux_info = {
            'cond_feat': cond_feat, 
            'curr_states': curr_states,
        }
        if include_class_free_cond:
            aux_info['non_cond_feat'] = non_cond_feat
        if self.use_map_feat_grid and self.map_encoder is not None:
            aux_info['map_grid_feat'] = map_grid_feat
            if include_class_free_cond:
                aux_info['map_grid_feat_non_cond'] = map_grid_feat_non_cond
            aux_info['raster_from_agent'] = raster_from_agent

        return aux_info

    def query_map_feats(self, x, map_grid_feat, raster_from_agent):
        '''
        - x : (B, T, D)
        - map_grid_feat : (B, C, H, W)
        - raster_from_agent: (B, 3, 3)
        '''
        B, T, _ = x.size()
        _, C, Hfeat, Wfeat = map_grid_feat.size()

        # unscale to agent coords
        pos_traj = self.descale_traj(x.detach())[:,:,:2]
        # convert to raster frame
        raster_pos_traj = transform_points_tensor(pos_traj, raster_from_agent)

        # scale to the feature map size
        _, H, W = self.input_image_shape
        xscale = Wfeat / W
        yscale = Hfeat / H
        raster_pos_traj[:,:,0] = raster_pos_traj[:,:,0] * xscale
        raster_pos_traj[:,:,1] = raster_pos_traj[:,:,1] * yscale

        # interpolate into feature grid
        feats_out = query_feature_grid(
                            raster_pos_traj,
                            map_grid_feat
                            )
        feats_out = feats_out.reshape((B, T, -1))
        return feats_out

    def get_state_and_action_from_data_batch(self, data_batch, chosen_inds=[]):
        '''
        Extract state and(or) action from the data_batch from data_batch

        Input:
            data_batch: dict
        Output:
            x: (batch_size, num_steps, len(chosen_inds)).
        '''
        if len(chosen_inds) == 0:
            chosen_inds = self.default_chosen_inds

        # NOTE: for predicted agent, history and future with always be fully available
        traj_state = torch.cat(
                (data_batch["target_positions"], data_batch["target_yaws"]), dim=2)

        traj_state_and_action = convert_state_to_state_and_action(traj_state, data_batch["curr_speed"], self.dt)

        return traj_state_and_action[..., chosen_inds]
    
    def convert_action_to_state_and_action(self, x_out, aux_info, scaled_input=True, descaled_output=False):
        '''
        Apply dynamics on input action trajectory to get state+action trajectory
        Input:
            x_out: (batch_size, num_steps, 2). scaled action trajectory
        Output:
            x_out: (batch_size, num_steps, 6). scaled state+action trajectory
        '''
        if scaled_input:
            x_out = self.descale_traj(x_out, [4, 5])
        
        x_out_state = unicyle_forward_dynamics(
            dyn_model=self.dyn,
            initial_states=aux_info['curr_states'],
            actions=x_out,
            step_time=self.dt,
            mode='parallel'
        )

        x_out_all = torch.cat([x_out_state, x_out], dim=-1)
        if scaled_input and not descaled_output:
            x_out_all = self.scale_traj(x_out_all, [0, 1, 2, 3, 4, 5])

        return x_out_all

    def scale_traj(self, target_traj_orig, chosen_inds=[]):
        '''
        - traj: B x T x D
        '''
        if len(chosen_inds) == 0:
            chosen_inds = self.default_chosen_inds
        add_coeffs = self.add_coeffs[chosen_inds][None,None] 
        div_coeffs = self.div_coeffs[chosen_inds][None,None] 

        device = target_traj_orig.get_device()
        dx_add = torch.tensor(add_coeffs, device=device)
        dx_div = torch.tensor(div_coeffs, device=device)

        target_traj = (target_traj_orig + dx_add) / dx_div

        return target_traj

    def descale_traj(self, target_traj_orig, chosen_inds=[]):
        '''
        - traj: B x T x D
        '''
        if len(chosen_inds) == 0:
            chosen_inds = self.default_chosen_inds
        add_coeffs = self.add_coeffs[chosen_inds][None,None] 
        div_coeffs = self.div_coeffs[chosen_inds][None,None] 

        device = target_traj_orig.get_device()
        dx_add = torch.tensor(add_coeffs, device=device)
        dx_div = torch.tensor(div_coeffs, device=device) 

        target_traj = target_traj_orig * dx_div - dx_add

        return target_traj

    
    def forward(self, data_batch: Dict[str, torch.Tensor], num_samp=1,
                    return_diffusion=False,
                    return_guidance_losses=False,
                    class_free_guide_w=0.0,
                    apply_guidance=True,
                    guide_clean=False) -> Dict[str, torch.Tensor]:
        use_class_free_guide = class_free_guide_w != 0.0
        aux_info = self.get_aux_info(data_batch, use_class_free_guide)
        
        cond_samp_out = self.conditional_sample(data_batch, 
                                                horizon=None,
                                                aux_info=aux_info,
                                                return_diffusion=return_diffusion,
                                                return_guidance_losses=return_guidance_losses,
                                                num_samp=num_samp,
                                                class_free_guide_w=class_free_guide_w,
                                                apply_guidance=apply_guidance,
                                                guide_clean=guide_clean)
        traj_init = cond_samp_out['pred_traj']
        diff_init = guide_losses = None
        if return_diffusion:
            diff_init = cond_samp_out['diffusion']
        if return_guidance_losses:
            guide_losses = cond_samp_out['guide_losses']

        traj = self.descale_traj(traj_init)
        if diff_init is not None:
            diff_steps = self.descale_traj(diff_init)
        else:
            diff_steps = None

        if self.diffuser_input_mode in ['state_and_action']:
            traj = traj[..., [0, 1, 3]]
        else:
            raise

        pred_positions = traj[..., :2]
        pred_yaws = traj[..., 2:3]

        out_dict = {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws},
        }
        if diff_steps is not None:
            out_dict["predictions"]["diffusion_steps"] = diff_steps
        if guide_losses is not None:
            out_dict["predictions"]["guide_losses"] = guide_losses
        if self.dyn is not None:
            out_dict["curr_states"] = aux_info['curr_states']

        return out_dict

    def compute_losses(self, data_batch):
        aux_info = self.get_aux_info(data_batch)
        target_traj = self.get_state_and_action_from_data_batch(data_batch)       

        x = self.scale_traj(target_traj)
                
        diffusion_loss, _ = self.loss(x, aux_info=aux_info)
        losses = OrderedDict(
            diffusion_loss=diffusion_loss,
        )
        return losses

    def get_loss_weights(self, action_weight, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        ## manually set a0 weight
        loss_weights[0, -self.action_dim:] = action_weight

        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#
    def predict_start_from_noise(self, x_t, t, noise, force_noise=False):
        if force_noise:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def predict_noise_from_start(self, x_t, t, x_start):
        return (
            extract(self.sqrt_recip_one_minus_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
            extract(self.sqrt_alphas_over_one_minus_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_start
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, aux_info={}, class_free_guide_w=0.0):
        t_inp = t

        x_model_in = x
        if self.use_map_feat_grid and self.map_encoder is not None:
            # get features from map and append to the trajectory
            map_feat_traj = self.query_map_feats(x_model_in.detach(),
                                                 aux_info['map_grid_feat'],
                                                 aux_info['raster_from_agent'])
            x_model_in = torch.cat([x_model_in, map_feat_traj], dim=-1)

        model_prediction = self.model(x_model_in, aux_info['cond_feat'], t_inp)

        if self.diffuser_input_mode == 'state_and_action':
            x_tmp = x[..., 4:].detach()
        else:
            raise

        if class_free_guide_w != 0.0:
            # now run non-cond once
            x_model_non_cond_in = x
            if self.use_map_feat_grid and self.map_encoder is not None:
                # get features from map and append to the trajectory
                map_feat_traj = self.query_map_feats(x_model_non_cond_in.detach(),
                                                    aux_info['map_grid_feat_non_cond'],
                                                    aux_info['raster_from_agent'])
                x_model_non_cond_in = torch.cat([x_model_non_cond_in, map_feat_traj], dim=-1)
            model_non_cond_prediction = self.model(x_model_non_cond_in, aux_info['non_cond_feat'], t_inp)

            # and combine to get actual model prediction (in noise space as in original paper)
            model_pred_noise = self.predict_noise_from_start(x_tmp, t=t, x_start=model_prediction)
            model_non_cond_pred_noise = self.predict_noise_from_start(x_tmp, t=t, x_start=model_non_cond_prediction)

            class_free_guide_noise = (1 + class_free_guide_w)*model_pred_noise - class_free_guide_w*model_non_cond_pred_noise

            model_prediction = self.predict_start_from_noise(x_tmp, t=t, noise=class_free_guide_noise, force_noise=True)

        x_recon = self.predict_start_from_noise(x_tmp, t=t, noise=model_prediction)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x_tmp, t=t)
        return model_mean, posterior_variance, posterior_log_variance, (x_recon, x_tmp, t)

    def guidance(self, x, data_batch, aux_info, num_samp=1,
                    return_grad_of=None):
        '''
        estimate the gradient of rule reward w.r.t. the input trajectory
        Input:
            x: [batch_size*num_samp, time_steps, feature_dim].  scaled input trajectory.
            data_batch: additional info.
            aux_info: additional info.
            return_grad_of: which variable to take gradient of guidance loss wrt, if not given,
                            takes wrt the input x.
        '''
        assert self.current_guidance is not None, 'Must instantiate guidance object before calling'
        bsize = int(x.size(0) / num_samp)
        num_t = x.size(1)
        with torch.enable_grad():
            # losses are applied on unscaled trajectories containing both states and actions
            if self.diffuser_input_mode in ['state_and_action']:
                # forward dynamics to get actions
                x_all = self.convert_action_to_state_and_action(x, aux_info, scaled_input=True, descaled_output=True)
            else:
                raise

            # compute losses and gradient
            x_loss = x_all.reshape((bsize, num_samp, num_t, 6))
            tot_loss, per_losses = self.current_guidance.compute_guidance_loss(x_loss, data_batch)
            # print(tot_loss)
            tot_loss.backward()
            guide_grad = x.grad if return_grad_of is None else return_grad_of.grad

            return guide_grad, per_losses

    @torch.no_grad()
    def p_sample(self, x, t, data_batch, aux_info={}, num_samp=1, class_free_guide_w=0.0, 
                 apply_guidance=True, guide_clean=False, eval_final_guide_loss=False):
        b, *_, device = *x.shape, x.device
        with_func = torch.no_grad
        if self.current_guidance is not None and apply_guidance and guide_clean:
            # will need to take grad wrt noisy input
            x = x.detach()
            x.requires_grad_()
            with_func = torch.enable_grad

        with with_func():
            # get prior mean and variance for next step
            model_mean, _, model_log_variance, q_posterior_in = self.p_mean_variance(x=x, t=t, aux_info=aux_info,
                                                                                    class_free_guide_w=class_free_guide_w)

        # no noise or guidance when t == 0
        #       i.e. use the mean of the distribution predicted at the final step rather than sampling.
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        noise = torch.randn_like(model_mean)
        sigma = (0.5 * model_log_variance).exp()

        # compute guidance
        guide_losses = None
        guide_grad = torch.zeros_like(model_mean)
        if self.current_guidance is not None and apply_guidance:
            if guide_clean:
                # want to guide the predicted clean traj from model, not the noisy one
                model_clean_pred = q_posterior_in[0]
                x_guidance = model_clean_pred
                return_grad_of = x
            else:
                x_guidance = model_mean.clone().detach()
                return_grad_of = x_guidance
                x_guidance.requires_grad_()

            guide_grad, guide_losses = self.guidance(x_guidance, data_batch, aux_info, num_samp=num_samp, return_grad_of=return_grad_of)

            if guide_clean and self.diffuser_input_mode == 'state_and_action':
                # only need the grad w.r.t noisy action
                guide_grad = guide_grad[..., [4,5]]

            # NOTE: empirally, scaling by the variance (sigma) seems to degrade results
            guide_grad = nonzero_mask * guide_grad #* sigma

        noise = nonzero_mask * sigma * noise

        if self.current_guidance is not None and guide_clean:
            # perturb clean trajectory
            guided_clean = q_posterior_in[0] - guide_grad
            # use the same noisy input again
            guided_x_t = q_posterior_in[1]
            # re-compute next step distribution with guided clean & noisy trajectories
            model_mean, _, _ = self.q_posterior(x_start=guided_clean,
                                                x_t=guided_x_t,
                                                t=q_posterior_in[2])
            # NOTE: variance is not dependent on x_start, so it won't change. Therefore, fine to use same noise.
            x_out = model_mean + noise
        else:
            x_out = model_mean - guide_grad + noise

        if self.current_guidance is not None and eval_final_guide_loss:
            # eval guidance loss one last time for filtering if desired
            #       (even if not applied during sampling)
            _, guide_losses = self.guidance(x_out.clone().detach().requires_grad_(), data_batch, aux_info, num_samp=num_samp)
        
        # convert action to state+action
        if self.diffuser_input_mode == 'state_and_action':
            x_out = self.convert_action_to_state_and_action(x_out, aux_info)
        
        return x_out, guide_losses

        
    @torch.no_grad()
    def p_sample_loop(self, shape, data_batch, num_samp,
                    aux_info={},
                    return_diffusion=False,
                    return_guidance_losses=False,
                    class_free_guide_w=0.0,
                    apply_guidance=True,
                    guide_clean=False):
        device = self.betas.device

        batch_size = shape[0]
        # sample from base distribution
        x = torch.randn(shape, device=device) # (B, N, T, D)

        x = TensorUtils.join_dimensions(x, begin_axis=0, end_axis=2) # B*N, T, D
        aux_info = TensorUtils.repeat_by_expand_at(aux_info, repeats=num_samp, dim=0)

        if self.current_guidance is not None and not apply_guidance:
            print('DIFFUSER: Note, not using guidance during sampling, only evaluating guidance loss at very end...')

        # convert action to state+action
        if self.diffuser_input_mode == 'state_and_action':
            x = self.convert_action_to_state_and_action(x[..., [4, 5]], aux_info)

        if return_diffusion: diffusion = [x]

        stride = 1 # NOTE: different from training time if > 1
        steps = [i for i in reversed(range(0, self.n_timesteps, stride))]
        for i in steps:
            timesteps = torch.full((batch_size*num_samp,), i, device=device, dtype=torch.long)
            
            x, guide_losses = self.p_sample(x, timesteps, data_batch,
                                            aux_info=aux_info,
                                            num_samp=num_samp,
                                            class_free_guide_w=class_free_guide_w,
                                            apply_guidance=apply_guidance,
                                            guide_clean=guide_clean,
                                            eval_final_guide_loss=(i == steps[-1]))
            

            if return_diffusion: diffusion.append(x)

        if guide_losses is not None:
            print('===== GUIDANCE LOSSES ======')
            for k,v in guide_losses.items():
                print('%s: %.012f' % (k, np.nanmean(v.cpu())))

        x = TensorUtils.reshape_dimensions(x, begin_axis=0, end_axis=1, target_dims=(batch_size, num_samp))

        out_dict = {'pred_traj' : x}
        if return_guidance_losses:
            out_dict['guide_losses'] = guide_losses
        if return_diffusion:
            diffusion = [TensorUtils.reshape_dimensions(cur_diff, begin_axis=0, end_axis=1, target_dims=(batch_size, num_samp))
                         for cur_diff in diffusion]
            out_dict['diffusion'] = torch.stack(diffusion, dim=3)

        return out_dict

    @torch.no_grad()
    def conditional_sample(self, data_batch, horizon=None, num_samp=1, class_free_guide_w=0.0, **kwargs):
        batch_size = data_batch['history_positions'].size()[0]
        horizon = horizon or self.horizon
        shape = (batch_size, num_samp, horizon, self.transition_dim)

        return self.p_sample_loop(shape, data_batch, num_samp, class_free_guide_w=class_free_guide_w, **kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise):        
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start_init, t, aux_info={}):
        noise_init = torch.randn_like(x_start_init)

        x_start = x_start_init
        noise = noise_init
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        t_inp = t

        if self.diffuser_input_mode == 'state_and_action':
            x_action_noisy = x_noisy[..., [4, 5]]
            x_noisy = self.convert_action_to_state_and_action(x_action_noisy, aux_info)

        if self.use_map_feat_grid and self.map_encoder is not None:
            # get features from map and append to the trajectory
            map_feat_traj = self.query_map_feats(x_noisy.detach(),
                                                 aux_info['map_grid_feat'],
                                                 aux_info['raster_from_agent'])
            x_noisy = torch.cat([x_noisy, map_feat_traj], dim=-1)

        noise = self.model(x_noisy, aux_info['cond_feat'], t_inp)

        if self.diffuser_input_mode == 'state_and_action':
            x_recon_action = self.predict_start_from_noise(x_action_noisy, t=t, noise=noise)
            x_recon = self.convert_action_to_state_and_action(x_recon_action, aux_info)
        else:
            x_recon = self.predict_start_from_noise(x_noisy, t=t, noise=noise)

        # Note: we convert noise into x_start for loss estimation since we need to apply forward dynamics
        loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, aux_info={}):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
            
        return self.p_losses(x, t, aux_info=aux_info)