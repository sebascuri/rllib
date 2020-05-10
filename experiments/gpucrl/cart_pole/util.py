"""Utilities for cartpole experiments."""

import torch
import torch.nn as nn


def termination(state, action, next_state=None):
    """Termination condition for environment."""
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state)
    if not isinstance(action, torch.Tensor):
        action = torch.tensor(action)

    return (torch.any(torch.abs(state) > 200, dim=-1) | torch.any(
        torch.abs(action) > 15, dim=-1))


class StateTransform(nn.Module):
    """Transform pendulum states to cos, sin, angular_velocity."""
    extra_dim = 1

    def forward(self, states_):
        """Transform state before applying function approximation."""
        position, angle, velocity, angular_velocity = torch.split(states_, 1, dim=-1)
        states_ = torch.cat((torch.cos(angle), torch.sin(angle), position, velocity,
                             angular_velocity),
                            dim=-1)
        return states_

    def inverse(self, states_):
        """Inverse transformation of states."""
        cos, sin, position, velocity, angular_velocity = torch.split(states_, 1, dim=-1)
        angle = torch.atan2(sin, cos)
        states_ = torch.cat((position, angle, velocity, angular_velocity), dim=-1)
        return states_