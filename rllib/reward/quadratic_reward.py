"""Model for quadratic reward."""

from .abstract_reward import AbstractReward
from gpytorch.distributions import Delta
from rllib.util.neural_networks import torch_quadratic


class QuadraticReward(AbstractReward):
    """Quadratic Reward Function."""

    def __init__(self, q, r):
        super().__init__()

        self.q = q
        self.r = r

    def forward(self, state, action):
        """See `abstract_reward.forward'."""
        state_cost = torch_quadratic(state, self.q)
        action_cost = torch_quadratic(action, self.r)
        return Delta(-(state_cost + action_cost).squeeze())
