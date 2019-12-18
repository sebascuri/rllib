"""Wrapper for Custom System Environments."""


from .abstract_environment import AbstractEnvironment


__all__ = ['SystemEnvironment']


class SystemEnvironment(AbstractEnvironment):
    """Wrapper for System Environments.

    Parameters
    ----------
    system: AbstractSystem
        underlying system
    initial_state: callable, optional
        callable that returns an initial state
    reward: callable, optional
        callable that, given state and action returns a rewards
    termination: callable, optional
        callable that checks if interaction should terminate.

    """

    def __init__(self, system, initial_state=None, reward=None, termination=None):
        super().__init__(
            dim_state=system.dim_state,
            dim_action=system.dim_action,
            dim_observation=system.dim_observation,
            action_space=system.action_space,
            observation_space=system.observation_space
        )
        self.reward = reward
        self.system = system
        self.termination = termination
        self._time = 0

        if initial_state is None:
            initial_state = self.system.observation_space.sample()
        if not callable(initial_state):
            self.initial_state = lambda: initial_state
        else:
            self.initial_state = initial_state

    def render(self, mode='human'):
        """See `AbstractEnvironment.render'."""
        self.system.render(mode=mode)

    def step(self, action):
        """See `AbstractEnvironment.step'."""
        self._time += 1
        state = self.system.state  # this might be noisy.
        reward = None
        if self.reward is not None:
            reward = self.reward(state, action)

        next_state = self.system.step(action)
        if self.termination is not None:
            done = self.termination(next_state)
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
