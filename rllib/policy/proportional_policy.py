"""Proportional policy implementation."""

import torch
import torch.nn as nn

from rllib.policy.nn_policy import NNPolicy
from rllib.util.neural_networks.utilities import to_torch


class ProportionalModule(nn.Module):
    """Proportional Module."""

    def __init__(self, gain, bias=None, offset=None, fixed=True):
        super().__init__()
        self.w = nn.Linear(
            in_features=gain.shape[1], out_features=gain.shape[0], bias=False
        )
        if bias is None and fixed:
            bias = torch.zeros_like(self.w.bias)
            self.w.bias = nn.Parameter(bias, requires_grad=not fixed)
        self.w.weight = nn.Parameter(gain, requires_grad=not fixed)
        self.offset = offset if offset is not None else torch.zeros(gain.shape[1])

    def forward(self, x):
        """Compute the output of the module."""
        out = self.w(x - self.offset)
        return out, torch.zeros_like(out)


class ProportionalPolicy(NNPolicy):
    """Proportional Policy implementation of AbstractPolicy base class.

    This policy will always return an action that is proportional to the state.
    It is only implemented for continuous action environments.

    """

    def __init__(
        self,
        gain,
        bias=None,
        offset=None,
        fixed=True,
        deterministic=True,
        *args,
        **kwargs
    ):
        super().__init__(deterministic=deterministic, layers=(), *args, **kwargs)
        gain = to_torch(gain)
        self.nn = ProportionalModule(gain=gain, bias=bias, offset=offset, fixed=fixed)

        if self.discrete_action:
            raise NotImplementedError("Actions can't be discrete.")

    @classmethod
    def default(cls, environment, gain=None, *args, **kwargs):
        """See Abstract Policy default initialization method."""
        if gain is None:
            gain = -torch.eye(environment.dim_action[0], environment.dim_state[0])
        return super().default(environment, gain=gain, *args, **kwargs)


if __name__ == "__main__":
    from rllib.environment.system_environment import SystemEnvironment
    from rllib.environment.systems.pitch_control import PitchControl

    env = SystemEnvironment(PitchControl())
    default_policy = ProportionalPolicy.default(env)
    print(default_policy.nn)

    state = env.reset()
    for i in range(10):
        action = default_policy(state)[0]
        print(state, action)
        next_state, reward, done, info = env.step(action)
        state = next_state
