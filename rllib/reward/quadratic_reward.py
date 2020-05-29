"""Model for quadratic reward."""

import torch

from rllib.util.neural_networks import torch_quadratic

from .abstract_reward import AbstractReward


class QuadraticReward(AbstractReward):
    """Quadratic Reward Function."""

    def __init__(self, q, r):
        super().__init__()

        self.q = q
        self.r = r

    def forward(self, state, action, next_state):
        """See `abstract_reward.forward'."""
        state_cost = torch_quadratic(state, self.q)
        action_cost = torch_quadratic(action, self.r)
        return -(state_cost + action_cost).squeeze(), torch.zeros(1)
