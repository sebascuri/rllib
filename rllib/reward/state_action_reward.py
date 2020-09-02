"""State Action Reward Functions."""
from abc import ABCMeta

import torch

from rllib.model import AbstractModel
from rllib.util.utilities import get_backend


class StateActionReward(AbstractModel, metaclass=ABCMeta):
    r"""Base class for state-action reward functions.

    The reward is computed as:
        ..math:: r = r_{state} + \alpha r_{action},

    where r_{state} is an environment dependent reward function (to be implemented),
    r_{action} is the action cost, and \alpha is set by `action_cost_ratio'.

    the action reward is given by:
       ..math:: r_{action} = - \sum_{i=1}^{d} a_i^2, in non-sparse environments.
       ..math:: r_{action} =  e^{-\sum_{i=1}^{d} (a_i/scale_i)^2} - 1 in sparse envs.

    Parameters
    ----------
    action_cost_ratio: float, optional (default = 0.1)
        action cost ratio that weights the action to state ratio.
    sparse: bool, optional (default = False).
        flag that indicates whether the reward is sparse or global.
    goal: Tensor, optional (default = None).
        Goal position, optional.
    action_scale: float, optional (default = 1.0).
        scale of action for sparse environments.
    """

    def __init__(
        self, action_cost_ratio=0.1, sparse=False, goal=None, action_scale=1.0
    ):
        super().__init__(
            goal=goal, dim_state=(), dim_action=self.dim_action, model_kind="rewards"
        )

        self.action_scale = action_scale
        self.action_cost_ratio = action_cost_ratio
        self.sparse = sparse

    def forward(self, state, action, next_state=None):
        """Get reward distribution for state, action, next_state."""
        reward_ctrl = self.action_reward(action)
        reward_state = self.state_reward(state, next_state)
        reward = reward_state + self.action_cost_ratio * reward_ctrl

        try:
            self._info.update(
                reward_state=reward_state.sum().item(),
                reward_ctrl=reward_ctrl.sum().item(),
            )
            return reward.type(torch.get_default_dtype()).unsqueeze(-1), torch.zeros(1)
        except AttributeError:
            return reward, torch.zeros(1)

    def action_reward(self, action):
        """Get reward that corresponds to action."""
        action = action[..., : self.dim_action[0]]  # get only true dimensions.
        bk = get_backend(action)
        if self.sparse:
            return bk.exp(-bk.square(action / self.action_scale).sum(-1)) - 1
        else:
            return -bk.square(action).sum(-1)

    def state_reward(self, state, next_state=None):
        """Get reward that corresponds to the states."""
        raise NotImplementedError
