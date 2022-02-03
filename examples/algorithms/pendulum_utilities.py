"""Utilities for Inverted Pendulum."""
import torch
from torch.distributions import MultivariateNormal

from rllib.model import AbstractModel
from rllib.reward.utilities import tolerance
from rllib.util.neural_networks.utilities import to_torch


class PendulumSparseReward(AbstractModel):
    """Reward for Inverted Pendulum."""

    def __init__(self, action_cost=0):
        super().__init__(dim_state=(2,), dim_action=(1,), model_kind="rewards")
        self.action_cost = action_cost
        self.reward_offset = 0

    def forward(self, state, action, next_state):
        """See `abstract_reward.forward'."""
        state, action = to_torch(state), to_torch(action)

        cos_angle = torch.cos(state[..., 0])
        velocity = state[..., 1]

        angle_tolerance = tolerance(cos_angle, lower=0.95, upper=1.0, margin=0.1)
        velocity_tolerance = tolerance(velocity, lower=-0.5, upper=0.5, margin=0.5)
        state_cost = angle_tolerance * velocity_tolerance

        action_tolerance = tolerance(action[..., 0], lower=-0.1, upper=0.1, margin=0.1)
        action_cost = self.action_cost * (action_tolerance - 1)

        cost = state_cost + action_cost

        return cost.unsqueeze(-1), torch.zeros(1)


class PendulumDenseReward(AbstractModel):
    """Reward for Inverted Pendulum."""

    def __init__(self, action_cost=0.0):
        super().__init__(dim_state=(2,), dim_action=(1,), model_kind="rewards")
        self.action_cost = action_cost
        self.reward_offset = 0

    def forward(self, state, action, next_state):
        """See `abstract_reward.forward'."""
        state, action = to_torch(state), to_torch(action)

        cos_angle = 1 - torch.cos(state[..., 0])
        state_cost = cos_angle ** 2
        action_cost = self.action_cost * (action ** 2).sum(-1)

        return -(action_cost + state_cost), torch.tensor(0.0)


class PendulumModel(AbstractModel):
    """Pendulum Model.

    Torch implementation of a pendulum model using euler forwards integration.
    """

    def __init__(
        self, mass, length, friction, step_size=1 / 80, noise: MultivariateNormal = None
    ):
        super().__init__(dim_state=(2,), dim_action=(1,))
        self.mass = mass
        self.length = length
        self.friction = friction
        self.step_size = step_size
        self.noise = noise

    def forward(self, state, action):
        """Get next-state distribution."""
        # Physical dynamics
        action = action.clamp(-1.0, 1.0)
        mass = self.mass
        gravity = 9.81
        length = self.length
        friction = self.friction
        inertia = mass * length ** 2
        dt = self.step_size

        angle, angular_velocity = torch.split(state, 1, dim=-1)
        for _ in range(1):
            x_ddot = (
                (gravity / length) * torch.sin(angle)
                + action * (1 / inertia)
                - (friction / inertia) * angular_velocity
            )

            angle = angle + dt * angular_velocity
            angular_velocity = angular_velocity + dt * x_ddot

        next_state = torch.cat((angle, angular_velocity), dim=-1)

        if self.noise is None:
            return next_state, torch.zeros(1)
        else:
            return next_state + self.noise.mean, self.noise.covariance_matrix
