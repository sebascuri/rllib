"""State Action Reward Functions."""
from abc import ABCMeta

import torch

from rllib.model import AbstractModel
from rllib.reward.utilities import tolerance


class StateActionReward(AbstractModel, metaclass=ABCMeta):
    r"""Base class for state-action reward functions.

    The reward is computed as:
        ..math:: r = r_{state} + \alpha r_{action},

    where r_{state} is an environment dependent reward function (to be implemented),
    r_{action} is the action cost, and \alpha is set by `ctrl_cost_weight'.

    the action reward is given by:
       ..math:: r_{action} = - \sum_{i=1}^{d} a_i^2, in non-sparse environments.
       ..math:: r_{action} =  e^{-\sum_{i=1}^{d} (a_i/scale_i)^2} - 1 in sparse envs.

    Parameters
    ----------
    ctrl_cost_weight: float, optional (default = 0.1)
        action cost ratio that weights the action to state ratio.
    sparse: bool, optional (default = False).
        flag that indicates whether the reward is sparse or global.
    goal: Tensor, optional (default = None).
        Goal position, optional.
    action_scale: float, optional (default = 1.0).
        scale of action for sparse environments.
    """

    def __init__(self, ctrl_cost_weight=0.1, sparse=False, goal=None, action_scale=1.0):
        super().__init__(
            goal=goal, dim_state=(), dim_action=self.dim_action, model_kind="rewards"
        )

        self.action_scale = action_scale
        self.ctrl_cost_weight = ctrl_cost_weight
        self.sparse = sparse

    def forward(self, state, action, next_state=None):
        """Get reward distribution for state, action, next_state."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.get_default_dtype())
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.get_default_dtype())

        reward_ctrl = self.action_reward(action)
        reward_state = self.state_reward(state, next_state)
        reward = reward_state + self.ctrl_cost_weight * reward_ctrl

        try:
            self._info.update(
                reward_state=reward_state.sum().item(),
                reward_ctrl=reward_ctrl.sum().item(),
            )
            reward = reward.type(torch.get_default_dtype()).unsqueeze(-1)
        except AttributeError:
            pass
        return reward, torch.zeros_like(reward).unsqueeze(-1)

    @staticmethod
    def action_sparse_reward(action):
        """Get action sparse reward."""
        return (tolerance(action, lower=-0.1, upper=0.1, margin=0.1) - 1).prod(dim=-1)

    @staticmethod
    def action_non_sparse_reward(action):
        """Get action non-sparse rewards."""
        return -(action ** 2).sum(-1)

    def action_reward(self, action):
        """Get reward that corresponds to action."""
        action = action[..., : self.dim_action[0]]  # get only true dimensions.
        if self.sparse:
            return self.action_sparse_reward(action)
        else:
            return self.action_non_sparse_reward(action / self.action_scale)

    def state_reward(self, state, next_state=None):
        """Get reward that corresponds to the states."""
        raise NotImplementedError
