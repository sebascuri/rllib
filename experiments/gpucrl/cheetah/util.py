"""Utilities for cartpole experiments."""

import torch


def termination(state, action, next_state=None):
    """Termination condition for environment."""
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state)
    if not isinstance(action, torch.Tensor):
        action = torch.tensor(action)

    return (torch.any(torch.abs(state) > 200, dim=-1) | torch.any(
        torch.abs(action) > 15, dim=-1))
