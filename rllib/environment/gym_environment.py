"""Wrapper for OpenAI-Gym Environments."""

from .abstract_environment import AbstractEnvironment
import gym


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
        self.env = gym.make(env_name).unwrapped
        self.env.seed(seed)
        try:
            dim_action = self.env.action_space.shape[0]
            num_actions = None
        except IndexError:
            dim_action = 1
            num_actions = self.env.action_space.n

        try:
            dim_state = self.env.observation_space.shape[0]
            num_states = None
        except IndexError:
            dim_state = 1
            # Add an extra final state as terminal state.
            num_states = self.env.observation_space.n + 1

        super().__init__(
            dim_action=dim_action,
            dim_state=dim_state,
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            num_actions=num_actions,
            num_states=num_states,
            num_observations=num_states
        )
        self._time = 0

    def step(self, action):
        """See `AbstractEnvironment.step'."""
        self._time += 1
        next_state, reward, done, info = self.env.step(action)
        if done and self.num_states is not None:
            next_state = self.num_states - 1
        return next_state, reward, done, info

    def render(self, mode='human'):
        """See `AbstractEnvironment.render'."""
        self.env.render(mode)

    def close(self):
        """See `AbstractEnvironment.close'."""
        self.env.close()

    def reset(self):
        """See `AbstractEnvironment.reset'."""
        self._time = 0
        return self.env.reset()

    @property
    def state(self):
        """See `AbstractEnvironment.state'."""
        try:
            return self.env.state
        except AttributeError:
            return self.env.s

    @state.setter
    def state(self, value):
        if hasattr(self.env, 'state'):
            self.env.state = value
        elif hasattr(self.env, 's'):
            self.env.s = value
        else:
            raise NotImplementedError('Strange state')

    @property
    def time(self):
        """See `AbstractEnvironment.time'."""
        return self._time

    @property
    def name(self):
        """Return class name."""
        return self.env.__class__.__name__
