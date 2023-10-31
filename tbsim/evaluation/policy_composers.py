"""A script for evaluating closed-loop simulation"""
from tbsim.algos.algos import (
    DiffuserTrafficModel,
)
from tbsim.utils.batch_utils import batch_utils
from tbsim.utils.config_utils import get_experiment_config_from_file
from tbsim.configs.base import ExperimentConfig

from tbsim.policies.wrappers import (
    PolicyWrapper,
)
from tbsim.utils.experiment_utils import get_checkpoint

class PolicyComposer(object):
    def __init__(self, eval_config, device, ckpt_root_dir="checkpoints/"):
        self.device = device
        self.ckpt_root_dir = ckpt_root_dir
        self.eval_config = eval_config
        self._exp_config = None

    def get_modality_shapes(self, exp_cfg: ExperimentConfig):
        return batch_utils().get_modality_shapes(exp_cfg)

    def get_policy(self):
        raise NotImplementedError

class Diffuser(PolicyComposer):
    """
    TRACE
    """
    def get_policy(self, policy=None):
        if policy is not None:
            assert isinstance(policy, DiffuserTrafficModel)
            policy_cfg = None
        else:
            policy_ckpt_path, policy_config_path = get_checkpoint(
                ckpt_key=self.eval_config.ckpt.policy.ckpt_key,
                ckpt_dir=self.eval_config.ckpt.policy.ckpt_dir,
                ckpt_root_dir=self.ckpt_root_dir,
            )
            policy_cfg = get_experiment_config_from_file(policy_config_path)
            policy = DiffuserTrafficModel.load_from_checkpoint(
                policy_ckpt_path,
                algo_config=policy_cfg.algo,
                modality_shapes=self.get_modality_shapes(policy_cfg)
            ).to(self.device).eval()
            policy_cfg = policy_cfg.clone()
        policy = PolicyWrapper.wrap_controller(
            policy,
            num_action_samples=self.eval_config.policy.num_action_samples,
            class_free_guide_w=self.eval_config.policy.class_free_guide_w,
            guide_as_filter_only=self.eval_config.policy.guide_as_filter_only,
            guide_with_gt=self.eval_config.policy.guide_with_gt,
            guide_clean=self.eval_config.policy.guide_clean,
        )

        return policy, policy_cfg