import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F

import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.metrics as Metrics
from tbsim.utils.batch_utils import batch_utils
from tbsim.policies.common import Action
from tbsim.models.trace import DiffuserModel
from tbsim.models.trace_helpers import EMA

from tbsim.utils.guidance_loss import choose_action_from_guidance, choose_action_from_gt

class DiffuserTrafficModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes, guidance_config=None):
        """
        Creates networks and places them into @self.nets.
        """
        super(DiffuserTrafficModel, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()

        if algo_config.diffuser_input_mode == 'state_and_action':
            # "Observations" are inputs to diffuser that are not outputs
            observation_dim = 4 # x, y, vel, yaw
            # "Actions" are inputs and outputs
            action_dim = 2 # acc, yawvel
            # "output" is final output of the entired denoising process
            output_dim = 2 # acc, yawvel
        else:
            raise
        
        self.cond_drop_map_p = algo_config.conditioning_drop_map_p
        self.cond_drop_neighbor_p = algo_config.conditioning_drop_neighbor_p
        min_cond_drop_p = min([self.cond_drop_map_p, self.cond_drop_neighbor_p])
        max_cond_drop_p = max([self.cond_drop_map_p, self.cond_drop_neighbor_p])
        assert min_cond_drop_p >= 0.0 and max_cond_drop_p <= 1.0
        self.use_cond = self.cond_drop_map_p < 1.0 and self.cond_drop_neighbor_p < 1.0 # no need for conditioning arch if always dropping
        self.cond_fill_val = algo_config.conditioning_drop_fill

        self.use_rasterized_map = algo_config.rasterized_map

        if self.use_cond:
            if self.cond_drop_map_p > 0:
                print('DIFFUSER: Dropping map input conditioning with p = %f during training...' % (self.cond_drop_map_p))
            if self.cond_drop_neighbor_p > 0:
                print('DIFFUSER: Dropping neighbor traj input conditioning with p = %f during training...' % (self.cond_drop_neighbor_p))

        self.nets["policy"] = DiffuserModel(
            rasterized_map=algo_config.rasterized_map,
            use_map_feat_global=algo_config.use_map_feat_global,
            use_map_feat_grid=algo_config.use_map_feat_grid,
            map_encoder_model_arch=algo_config.map_encoder_model_arch,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            map_feature_dim=algo_config.map_feature_dim,
            map_grid_feature_dim=algo_config.map_grid_feature_dim,
            hist_num_frames=algo_config.history_num_frames+1, # the current step is concat to the history
            hist_feature_dim=algo_config.history_feature_dim,
            cond_feature_dim=algo_config.cond_feat_dim,
            diffuser_model_arch=algo_config.diffuser_model_arch,
            horizon=algo_config.horizon,
            observation_dim=observation_dim, 
            action_dim=action_dim,
            output_dim=output_dim,
            n_timesteps=algo_config.n_diffusion_steps,
            loss_type=algo_config.loss_type, 
            action_weight=algo_config.action_weight, 
            loss_discount=algo_config.loss_discount, 
            dim_mults=algo_config.dim_mults,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            base_dim=algo_config.base_dim,
            diffuser_input_mode=algo_config.diffuser_input_mode,
            use_conditioning=self.use_cond,
            cond_fill_value=self.cond_fill_val,
            diffuser_norm_info=algo_config.diffuser_norm_info,
            agent_hist_norm_info=algo_config.agent_hist_norm_info,
            neighbor_hist_norm_info=algo_config.neighbor_hist_norm_info,
            dt=algo_config.step_time,
        )

        # set up initial guidance
        if guidance_config is not None:
            self.set_guidance(guidance_config)

        # set up EMA
        self.use_ema = algo_config.use_ema
        if self.use_ema:
            print('DIFFUSER: using EMA... val and get_action will use ema model')
            self.ema = EMA(algo_config.ema_decay)
            self.ema_policy = copy.deepcopy(self.nets["policy"])
            self.ema_policy.requires_grad_(False)
            self.ema_update_every = algo_config.ema_step
            self.ema_start_step = algo_config.ema_start_step
            self.reset_parameters()

        self.cur_train_step = 0

    @property
    def checkpoint_monitor_keys(self):
        if self.use_ema:
            return {"valLoss": "val/ema_losses_diffusion_loss"}
        else:
            return {"valLoss": "val/losses_diffusion_loss"}

    def forward(self, obs_dict, num_samp=1, class_free_guide_w=0.0, guide_as_filter_only=False, guide_clean=False):
        cur_policy = self.nets["policy"]
        # this function is only called at validation time, so use ema
        if self.use_ema:
            cur_policy = self.ema_policy
        return cur_policy(obs_dict, num_samp,
                                   return_diffusion=True,
                                   return_guidance_losses=True,
                                   class_free_guide_w=class_free_guide_w,
                                   apply_guidance=(not guide_as_filter_only),
                                   guide_clean=guide_clean)["predictions"]

    def _compute_metrics(self, pred_batch, data_batch):
        metrics = {}
        predictions = pred_batch["predictions"]
        preds = TensorUtils.to_numpy(predictions["positions"])
        gt = TensorUtils.to_numpy(data_batch["target_positions"])
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"])

        # compute ADE & FDE based on trajectory samples
        sample_preds = preds
        conf = np.ones(sample_preds.shape[0:2]) / float(sample_preds.shape[1])
        metrics["ego_avg_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()
        metrics["ego_avg_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()

        # compute diversity scores based on trajectory samples
        metrics["ego_avg_APD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_APD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "max").mean()
        metrics["ego_avg_FPD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_FPD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "max").mean()

        return metrics

    def reset_parameters(self):
        self.ema_policy.load_state_dict(self.nets["policy"].state_dict())

    def step_ema(self, step):
        if step < self.ema_start_step:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_policy, self.nets["policy"])

    def training_step_end(self, batch_parts):
        self.cur_train_step += 1

    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number (relative to the CURRENT epoch) - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        if self.use_ema and self.cur_train_step % self.ema_update_every == 0:
            self.step_ema(self.cur_train_step)

        batch = batch_utils().parse_batch(batch)
        
        # drop out conditioning if desired
        if self.use_cond:
            if self.use_rasterized_map:
                num_sem_layers = batch['maps'].size(1)
                if self.cond_drop_map_p > 0:
                    drop_mask = torch.rand((batch["image"].size(0))) < self.cond_drop_map_p
                    # only fill the last num_sem_layers as these correspond to semantic map
                    batch["image"][drop_mask, -num_sem_layers:] = self.cond_fill_val

            if self.cond_drop_neighbor_p > 0:
                # drop actual neighbor trajectories instead
                # set availability to False, will be zeroed out in model
                B = batch["all_other_agents_history_availabilities"].size(0)
                drop_mask = torch.rand((B)) < self.cond_drop_neighbor_p
                batch["all_other_agents_history_availabilities"][drop_mask] = 0

        # diffuser only take the data to estimate loss
        losses = self.nets["policy"].compute_losses(batch)

        total_loss = 0.0
        for lk, l in losses.items():
            losses[lk] = l * self.algo_config.loss_weights[lk]
            total_loss += losses[lk]

        for lk, l in losses.items():
            self.log("train/losses_" + lk, l)

        return {
            "loss": total_loss,
            "all_losses": losses,
        }

    def validation_step(self, batch, batch_idx):
        cur_policy = self.nets["policy"]

        batch = batch_utils().parse_batch(batch)
        
        losses = TensorUtils.detach(cur_policy.compute_losses(batch))
        
        pout = cur_policy(batch,
                        num_samp=self.algo_config.diffuser.num_eval_samples,
                        return_diffusion=False,
                        return_guidance_losses=False)
        metrics = self._compute_metrics(pout, batch)
        return_dict =  {"losses": losses, "metrics": metrics}

        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
            ema_losses = TensorUtils.detach(cur_policy.compute_losses(batch))
            pout = cur_policy(batch,
                        num_samp=self.algo_config.diffuser.num_eval_samples,
                        return_diffusion=False,
                        return_guidance_losses=False)
            ema_metrics = self._compute_metrics(pout, batch)
            return_dict["ema_losses"] = ema_losses
            return_dict["ema_metrics"] = ema_metrics

        return return_dict

    def validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)
        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)

        if self.use_ema:
            for k in outputs[0]["ema_losses"]:
                m = torch.stack([o["ema_losses"][k] for o in outputs]).mean()
                self.log("val/ema_losses_" + k, m)
            for k in outputs[0]["ema_metrics"]:
                m = np.stack([o["ema_metrics"][k] for o in outputs]).mean()
                self.log("val/ema_metrics_" + k, m)

    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params["policy"]
        return optim.Adam(
            params=self.nets["policy"].parameters(),
            lr=optim_params["learning_rate"]["initial"],
        )

    def get_action(self, obs_dict,
                    num_action_samples=1,
                    class_free_guide_w=0.0, 
                    guide_as_filter_only=False,
                    guide_with_gt=False,
                    guide_clean=False,
                    **kwargs):
        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy

        cur_policy.eval()

        # update with current "global" timestep
        cur_policy.update_guidance(global_t=kwargs['step_index'])

        preds = self(obs_dict,
                    num_samp=num_action_samples,
                    class_free_guide_w=class_free_guide_w,
                    guide_as_filter_only=guide_as_filter_only,
                    guide_clean=guide_clean) # [B, N, T, 2]
        B, N, T, _ = preds["positions"].size()

        # arbitrarily use the first sample as the action by default
        act_idx = torch.zeros((B), dtype=torch.long, device=preds["positions"].device) 
        if guide_with_gt and "target_positions" in obs_dict:
            act_idx = choose_action_from_gt(preds, obs_dict)
        elif cur_policy.current_guidance is not None:
            guide_losses = preds.pop("guide_losses", None)                
            act_idx = choose_action_from_guidance(preds, obs_dict, cur_policy.current_guidance.guide_configs, guide_losses)
                    
        action_preds = TensorUtils.map_tensor(preds, lambda x: x[torch.arange(B), act_idx])  

        info = dict(
            action_samples=Action(
                positions=preds["positions"],
                yaws=preds["yaws"]
            ).to_dict(),
            diffusion_steps={
                'traj' : action_preds["diffusion_steps"] # full state for the first sample
            },
        )
        action = Action(
            positions=action_preds["positions"],
            yaws=action_preds["yaws"]
        )

        return action, info

    def set_guidance(self, guidance_config, example_batch=None):
        '''
        Resets the test-time guidance functions to follow during prediction.
        '''
        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.set_guidance(guidance_config, example_batch)

    def clear_guidance(self):
        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.clear_guidance()