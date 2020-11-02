"""Mini-Atari Environment Wrapper."""

from importlib import import_module

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from gym import Env
from gym.spaces import Box, Discrete


class MiniAtariEnv(Env):
    """Mini Atari Environments for interesting MDPs.

    References
    ----------
    Young, K., & Tian, T. (2019).
    Minatar: An atari-inspired testbed for thorough and reproducible reinforcement
    learning experiments. https://github.com/kenjyoung/MinAtar

    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self, env_name, seed=None, sticky_action_prob=0.1, difficulty_ramping=True
    ):
        env_module = import_module("minatar.environments." + env_name)
        self._env = env_module.Env(ramping=difficulty_ramping, seed=seed)
        self.last_action = 0
        self.sticky_action_prob = sticky_action_prob
        self.action_space = Discrete(6)
        dim_state = self._get_obs().shape
        self.observation_space = Box(np.zeros(dim_state), np.ones(dim_state))
        self.n_channels = dim_state[0]
        self.ax = None

        color_map = sns.color_palette("cubehelix", self.n_channels)
        color_map.insert(0, (0, 0, 0))
        self.color_map = colors.ListedColormap(color_map)

        self.norm = colors.BoundaryNorm(
            [i for i in range(self.n_channels + 2)], self.n_channels + 1
        )

    def step(self, action):
        """Step environment."""
        reward, done = self._env.act(action)
        next_state = self._get_obs()
        return next_state, reward, done, {}

    def _get_obs(self):
        return self._env.state().transpose((2, 0, 1))

    def reset(self):
        """See `gym.Env.reset()'."""
        self._env.reset()
        return self._get_obs()

    def render(self, mode="human"):
        """Render the state."""
        if self.ax is None:
            _, self.ax = plt.subplots(1, 1)

        state = self._env.state()
        img = (
            np.amax(state * np.reshape(np.arange(self.n_channels) + 1, (1, 1, -1)), 2)
            + 0.5
        )
        plt.cla()
        self.ax.imshow(img, cmap=self.color_map, norm=self.norm, interpolation="none")
        plt.show(block=False)
        plt.pause(0.1 / 1000)
        if mode == "rgby_array":
            return img

    def close(self):
        """Close environment."""
        plt.close()
