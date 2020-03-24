"""Model for pendulum reward."""

import torch

from .abstract_reward import AbstractReward
from .utilities import tolerance


class PendulumReward(AbstractReward):
    """Reward for Inverted Pendulum."""

    def forward(self, state, action):
        """See `abstract_reward.forward'."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.get_default_dtype())
        cos_angle = torch.cos(state[..., 0])
        velocity = state[..., 1]

        angle_tolerance = tolerance(cos_angle, lower=0.95, upper=1., margin=0.1)
        velocity_tolerance = tolerance(velocity, lower=-.5, upper=0.5, margin=0.5)
        return angle_tolerance * velocity_tolerance, torch.zeros(1)
