import torch
from typing import Tuple, Dict

import tbsim.utils.tensor_utils as TensorUtils
from tbsim.policies.common import Action
from tbsim.policies.base import Policy

class GTNaNPolicy(Policy):
    '''
    Dummy policy to return GT action from data. If GT is non available fills
    in with nans (instead of 0's as above).
    '''
    def __init__(self, device):
        super(GTNaNPolicy, self).__init__(device)

    def eval(self):
        pass

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        invalid_mask = ~obs["target_availabilities"]
        gt_pos = TensorUtils.to_torch(obs["target_positions"], device=self.device)
        gt_pos[invalid_mask] = torch.nan
        gt_yaw = TensorUtils.to_torch(obs["target_yaws"], device=self.device)
        gt_yaw[invalid_mask] = torch.nan
        action = Action(
            positions=gt_pos,
            yaws=gt_yaw,
        )
        return action, {}