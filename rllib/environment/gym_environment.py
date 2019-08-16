"""Wrapper for OpenAI-Gym Environments."""

from .abstract_environment import AbstractEnvironment
import gym


__all__ = ['GymEnvironment']


class GymEnvironment(AbstractEnvironment):
    """Wrapper for OpenAI-Gym Environments.

    Parameters
    ----------
    env_name: str
        environment name
    seed: int, optional
        random seed to initialize environment.

    """

    def __init__(self, env_name, seed=None, max_steps=None):
        self._env = gym.make(env_name)
        self._env.seed(seed)
        try:
            dim_action = self._env.action_space.shape[0]
            num_actions = None
        except IndexError:
            dim_action = 1
            num_actions = self._env.action_space.n

        try:
            dim_state = self._env.observation_space.shape[0]
            num_states = None
        except IndexError:
            dim_state = 1
            num_states = self._env.observation_space.n

        super().__init__(
            dim_action=dim_action,
            dim_state=dim_state,
            action_space=self._env.action_space,
            observation_space=self._env.observation_space,
            num_actions=num_actions,
            num_observations=num_states
        )
        self._time = 0
        self.max_steps = max_steps if max_steps else float('Inf')

    @property
    def state(self):
        return self._env.state

    @state.setter
    def state(self, value):
        self._env.state = value

    @property
    def time(self):
        return self._time

    def step(self, action):
        self._time += 1
        state, reward, done, info = self._env.step(action)
        done |= self.time >= self.max_steps
        return state, reward, done, info

    def reset(self):
        self._time = 0
        return self._env.reset()
