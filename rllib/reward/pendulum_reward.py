"""Model for pendulum reward."""

import torch

from .abstract_reward import AbstractReward
from .utilities import tolerance


class PendulumReward(AbstractReward):
    """Reward for Inverted Pendulum."""

    def __init__(self, action_cost_ratio=0):
        super().__init__()
        self.action_cost_ratio = action_cost_ratio

    def forward(self, state, action):
        """See `abstract_reward.forward'."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.get_default_dtype())
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.get_default_dtype())

        cos_angle = torch.cos(state[..., 0])
        velocity = state[..., 1]

        angle_tolerance = tolerance(cos_angle, lower=0.95, upper=1., margin=0.1)
        velocity_tolerance = tolerance(velocity, lower=-.5, upper=0.5, margin=0.5)
        state_cost = angle_tolerance * velocity_tolerance

        action_tolerance = tolerance(action[..., 0], lower=-0.1, upper=0.1, margin=0.1)
        action_cost = self.action_cost_ratio * (action_tolerance-1)

        cost = state_cost + action_cost

        return cost, torch.zeros(1)
