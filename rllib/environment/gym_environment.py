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

    def __init__(self, env_name, seed=None):
        self._env = gym.make(env_name).unwrapped
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
            num_states=num_states,
            num_observations=num_states
        )
        self._time = 0

    def step(self, action):
        """See `AbstractEnvironment.step'."""
        self._time += 1
        state, reward, done, info = self._env.step(action)
        return state, reward, done, info

    def render(self, mode='human'):
        """See `AbstractEnvironment.render'."""
        self._env.render(mode)

    def close(self):
        """See `AbstractEnvironment.close'."""
        self._env.close()

    def reset(self):
        """See `AbstractEnvironment.reset'."""
        self._time = 0
        return self._env.reset()

    @property
    def state(self):
        """See `AbstractEnvironment.state'."""
        try:
            return self._env.state
        except AttributeError:
            return self._env.s

    @state.setter
    def state(self, value):
        if hasattr(self._env, 'state'):
            self._env.state = value
        elif hasattr(self._env, 's'):
            self._env.s = value
        else:
            raise NotImplementedError('Strange state')

    @property
    def time(self):
        """See `AbstractEnvironment.time'."""
        return self._time

    @property
    def name(self):
        """Return class name."""
        return self._env.__class__.__name__
