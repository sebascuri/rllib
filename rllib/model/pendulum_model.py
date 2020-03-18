"""Implementation of a Pendulum Model."""
from .abstract_model import AbstractModel
import torch
from torch.distributions import MultivariateNormal


class PendulumModel(AbstractModel):
    """Pendulum Model.

    Torch implementation of a pendulum model using euler forwards integration.
    """

    def __init__(self, mass, length, friction, step_size=1 / 80,
                 noise: MultivariateNormal = None):
        super().__init__(dim_state=2, dim_action=1)
        self.mass = mass
        self.length = length
        self.friction = friction
        self.step_size = step_size
        self.noise = noise

    def forward(self, state, action):
        """Get next-state distribution."""
        # Physical dynamics
        action = torch.clamp(action, -1., 1.)
        mass = self.mass
        gravity = 9.81
        length = self.length
        friction = self.friction
        inertia = mass * length ** 2
        dt = self.step_size

        angle, angular_velocity = torch.split(state, 1, dim=-1)
        # cos_angle, sin_angle, angular_velocity = torch.split(state, 1, dim=-1)
        # angle = torch.atan2(sin_angle, cos_angle)
        for _ in range(1):
            x_ddot = ((gravity / length) * torch.sin(angle)
                      + action * (1 / inertia)
                      - (friction / inertia) * angular_velocity)

            angle = angle + dt * angular_velocity
            angular_velocity = angular_velocity + dt * x_ddot

        next_state = (torch.cat((angle, angular_velocity), dim=-1))
        # next_state = (
        # torch.cat((torch.cos(angle), torch.sin(angle), angular_velocity), dim=-1))

        if self.noise is None:
            return next_state, torch.zeros(1)
        else:
            return next_state + self.noise.mean, self.noise.covariance_matrix
