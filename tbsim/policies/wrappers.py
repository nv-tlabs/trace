import torch
from typing import Tuple, Dict

from tbsim.policies.common import Action, Plan, RolloutAction

class PolicyWrapper(object):
    """A convenient wrapper for specifying run-time keyword arguments"""

    def __init__(self, model, get_action_kwargs=None, get_plan_kwargs=None):
        self.model = model
        self.device = model.device
        self.action_kwargs = get_action_kwargs
        self.plan_kwargs = get_plan_kwargs

    def eval(self):
        self.model.eval()

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        return self.model.get_action(obs, **self.action_kwargs, **kwargs)

    def get_plan(self, obs, **kwargs) -> Tuple[Plan, Dict]:
        return self.model.get_plan(obs, **self.plan_kwargs, **kwargs)

    @classmethod
    def wrap_controller(cls, model, **kwargs):
        return cls(model=model, get_action_kwargs=kwargs)

    @classmethod
    def wrap_planner(cls, model, **kwargs):
        return cls(model=model, get_plan_kwargs=kwargs)


class RolloutWrapper(object):
    """A wrapper policy that can (optionally) control both ego and other agents in a scene"""

    def __init__(self, ego_policy=None, agents_policy=None, pass_agent_obs=True):
        self.device = ego_policy.device if agents_policy is None else agents_policy.device
        self.ego_policy = ego_policy
        self.agents_policy = agents_policy
        self.pass_agent_obs = pass_agent_obs

    def eval(self):
        self.ego_policy.eval()
        self.agents_policy.eval()

    def get_action(self, obs, step_index) -> RolloutAction:
        ego_action = None
        ego_action_info = None
        agents_action = None
        agents_action_info = None
        if self.ego_policy is not None:
            assert obs["ego"] is not None
            with torch.no_grad():
                if self.pass_agent_obs:
                    ego_action, ego_action_info = self.ego_policy.get_action(
                        obs["ego"], step_index = step_index,agent_obs = obs["agents"])
                else:
                    ego_action, ego_action_info = self.ego_policy.get_action(
                        obs["ego"], step_index = step_index)
        if self.agents_policy is not None:
            assert obs["agents"] is not None
            with torch.no_grad():
                agents_action, agents_action_info = self.agents_policy.get_action(
                    obs["agents"], step_index = step_index)
        return RolloutAction(ego_action, ego_action_info, agents_action, agents_action_info)