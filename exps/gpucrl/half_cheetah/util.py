"""Utilities for cartpole experiments."""

import torch
import torch.nn as nn


class StateTransform(nn.Module):
    """Transform pendulum states to cos, sin, angular_velocity."""
    extra_dim = 1

    def forward(self, states):
        """Transform state before applying function approximation."""
        angles = states[..., 2:3]
        states_ = torch.cat(
            (states[..., :2], torch.cos(angles), torch.sin(angles), states[..., 3:]),
            dim=-1)
        return states_

    def inverse(self, states):
        """Inverse transformation of states."""
        cos, sin = states[..., 2:3], states[..., 3:4]
        angle = torch.atan2(sin, cos)
        states_ = torch.cat((states[..., :2], angle, states[..., 4:]), dim=-1)
        return states_
