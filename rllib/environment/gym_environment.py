"""Wrapper for OpenAI-Gym Environments."""

import gym
import gym.wrappers
import numpy as np
import torch

from .abstract_environment import AbstractEnvironment
from .utilities import parse_space


class GymEnvironment(AbstractEnvironment):
    """Wrapper for OpenAI-Gym Environments.

    Parameters
    ----------
    env_name: str
        environment name
    seed: int, optional
        random seed to initialize environment.

    """

    def __init__(self, env_name, seed=None, **kwargs):
        env = gym.make(env_name, **kwargs)
        if isinstance(env, gym.wrappers.TimeLimit) and not kwargs.get(
            "episodic", False
        ):
            env = env.unwrapped
        self.env = env
        self.env.seed(seed)
        self.env_name = env_name

        dim_action, num_actions = parse_space(self.env.action_space)
        dim_state, num_states = parse_space(self.env.observation_space)
        if num_states > -1:
            num_states += 1  # Add a terminal state.

        super().__init__(
            dim_action=dim_action,
            dim_state=dim_state,
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            num_actions=num_actions,
            num_states=num_states,
            dim_reward=self._get_dim_reward(),
            num_observations=num_states,
        )
        self._time = 0
        self.metadata = self.env.metadata

    def _get_dim_reward(self):
        if hasattr(self.env, "dim_reward"):
            return self.env.dim_reward
        else:
            return (1,)

    def _reset(self):
        dim_action, num_actions = parse_space(self.env.action_space)
        dim_state, num_states = parse_space(self.env.observation_space)
        if num_states > -1:
            num_states += 1  # Add a terminal state.

        super().__init__(
            dim_action=dim_action,
            dim_state=dim_state,
            dim_reward=self._get_dim_reward(),
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            num_actions=num_actions,
            num_states=num_states,
            num_observations=num_states,
        )
        self._time = 0

    def add_wrapper(self, wrapper, **kwargs):
        """Add a wrapper for the environment."""
        self.env = wrapper(self.env, **kwargs)
        self._reset()

    def pop_wrapper(self):
        """Pop last wrapper."""
        self.env = self.env.env
        self._reset()

    def step(self, action):
        """See `AbstractEnvironment.step'."""
        next_state, reward, done, info = self.env.step(action)
        if self.num_states > 0 and done:  # Move to terminal state.
            next_state = self.num_states - 1
        self._time += 1
        if isinstance(reward, torch.Tensor):
            if reward.shape[-1] != self.dim_reward[0]:
                reward = reward.unsqueeze(-1).repeat_interleave(self.dim_reward[0], -1)
        else:
            reward = np.atleast_1d(reward)
            if reward.shape[-1] != self.dim_reward[0]:
                reward = np.tile(reward, (self.dim_reward[0], 1)).T
        return next_state, reward, done, info

    def render(self, mode="human"):
        """See `AbstractEnvironment.render'."""
        return self.env.render(mode)

    def close(self):
        """See `AbstractEnvironment.close'."""
        self.env.close()

    def reset(self):
        """See `AbstractEnvironment.reset'."""
        self._time = 0
        return self.env.reset()

    @property
    def goal(self):
        """Return current goal of environment."""
        if hasattr(self.env, "goal"):
            return self.env.goal
        return None

    @property
    def state(self):
        """See `AbstractEnvironment.state'."""
        if hasattr(self.env, "state"):
            return self.env.state
        elif hasattr(self.env, "s"):
            return self.env.s
        else:
            raise NotImplementedError("Strange state")

    @state.setter
    def state(self, value):
        if hasattr(self.env, "set_state"):
            if hasattr(self.env, "sim"):  # mujocopy environments.
                self.env.set_state(
                    value[: len(self.env.sim.data.qpos)],
                    value[len(self.env.sim.data.qpos) :],
                )
            else:
                self.env.set_state(value)
        elif hasattr(self.env, "state"):
            self.env.state = value
        elif hasattr(self.env, "s"):
            self.env.s = value
        else:
            raise NotImplementedError("Strange state")

    @property
    def time(self):
        """See `AbstractEnvironment.time'."""
        return self._time

    @property
    def name(self):
        """Return class name."""
        return self.env_name
