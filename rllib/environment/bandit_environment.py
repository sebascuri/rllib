"""Bandit Environment."""

import numpy as np
from gym import spaces

from rllib.util.utilities import tensor_to_distribution

from .abstract_environment import AbstractEnvironment


class BanditEnvironment(AbstractEnvironment):
    """Bandit Environment.

    Parameters
    ----------
    reward: reward function of bandit.
    num_actions: number of actions in domain.
    x_min: minimum per-coordinate constraint of continuous domain.
    x_max: minimum per-coordinate constraint of continuous domain.

    Notes
    -----
    If num_actions is None, then the domain is continuous and x_min and x_max must be
    set.
    """

    def __init__(self, reward, num_actions=None, x_min=None, x_max=None):
        observation_space = spaces.Discrete(1)  # it has only one space

        if num_actions:
            action_space = spaces.Discrete(num_actions)
            dim_action = 1
        else:
            action_space = spaces.Box(x_min, x_max, dtype=np.float32)
            dim_action = action_space.shape[0]

        self.reward = reward
        self.t = 0

        super().__init__(
            dim_state=(),
            num_states=1,
            dim_action=dim_action,
            num_actions=num_actions,
            observation_space=observation_space,
            action_space=action_space,
        )

    def step(self, action):
        """Get reward of a given action."""
        self.t += 1
        reward_distribution = tensor_to_distribution(
            self.reward(self.state, action, None)
        )
        return self.state, reward_distribution.sample().numpy(), False, {}

    def reset(self):
        """Reset time counter to zero."""
        self.t = 0
        return self.state

    @property
    def state(self):
        """Get system state."""
        return self.observation_space.sample()

    @state.setter
    def state(self, value):
        """Set system state."""
        pass

    @property
    def time(self):
        """Get time."""
        return self.t
