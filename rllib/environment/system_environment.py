"""Wrapper for Custom System Environments."""
from abc import ABCMeta, abstractmethod

import torch
from gym import Env

from rllib.util.utilities import tensor_to_distribution

from .abstract_environment import AbstractEnvironment, EnvironmentBuilder


class SystemEnvironment(AbstractEnvironment, Env):
    """Wrapper for System Environments.

    Parameters
    ----------
    system: AbstractSystem
        underlying system
    initial_state: callable, optional
        callable that returns an initial state
    reward: callable, optional
        callable that, given state and action returns a rewards
    termination_model: callable, optional
        callable that checks if interaction should terminate.

    """

    def __init__(self, system, initial_state=None, reward=None, termination_model=None):
        super().__init__(
            dim_state=system.dim_state,
            dim_action=system.dim_action,
            dim_observation=system.dim_observation,
            action_space=system.action_space,
            observation_space=system.observation_space,
            dim_reward=reward.dim_reward if reward is not None else (1,),
        )
        self.reward = reward
        self.system = system
        self.termination_model = termination_model
        self._time = 0

        if initial_state is None:
            initial_state = self.system.observation_space.sample

        if not callable(initial_state):
            self.initial_state = lambda: initial_state
        else:
            self.initial_state = initial_state

    def render(self, mode="human"):
        """See `AbstractEnvironment.render'."""
        return self.system.render(mode=mode)

    def step(self, action):
        """See `AbstractEnvironment.step'."""
        self._time += 1
        state = self.system.state  # this might be noisy.
        reward = torch.tensor([float("nan")])
        if self.reward is not None:
            reward = tensor_to_distribution(self.reward(state, action, None)).sample()

        next_state = self.system.step(action)
        if self.termination_model is not None:
            done = (
                tensor_to_distribution(
                    self.termination_model(state, action, next_state)
                )
                .sample()
                .squeeze(-1)
            )
        else:
            done = False

        return next_state, reward, done, {}

    def reset(self):
        """See `AbstractEnvironment.reset'."""
        initial_state = self.initial_state()
        self._time = 0
        return self.system.reset(initial_state)

    @property
    def state(self):
        """See `AbstractEnvironment.state'."""
        return self.system.state

    @state.setter
    def state(self, value):
        self.system.state = value

    @property
    def time(self):
        """See `AbstractEnvironment.time'."""
        return self._time

    @property
    def name(self):
        """Return class name."""
        return self.system.__class__.__name__


class SystemEnvironmentBuilder(EnvironmentBuilder, metaclass=ABCMeta):
    """System Environment default Builder."""

    def create_environment(self):
        """Create environment."""
        return SystemEnvironment(
            system=self.get_system_model(),
            initial_state=self.initial_distribution_fn(),
            reward=self.get_reward_model(),
            termination_model=self.get_termination_model(),
        )

    @abstractmethod
    def get_system_model(self):
        """Get dynamical model."""
        raise NotImplementedError

    def initial_distribution_fn(self):
        """Get Initial Distribution Sample function."""
        return None
