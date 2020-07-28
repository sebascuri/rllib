"""Model for quadratic reward."""

import torch

from rllib.util.neural_networks import torch_quadratic

from .abstract_reward import AbstractReward


class QuadraticReward(AbstractReward):
    """Quadratic Reward Function."""

    def __init__(self, q, r, goal=None):
        if goal is None:
            goal = torch.zeros(q.shape[-1])
        super().__init__(goal=goal)

        self.q = q
        self.r = r

    def forward(self, state, action, next_state):
        """See `abstract_reward.forward'."""
        state_cost = torch_quadratic(state - self.goal, self.q)
        action_cost = torch_quadratic(action, self.r)
        return -(state_cost + action_cost).squeeze(-1), torch.zeros(1)
