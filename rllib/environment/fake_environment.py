"""A fake environment is an environment that is simulated with a model."""
from rllib.util.utilities import sample_model
from rllib.util.neural_networks.utilities import to_torch
from .abstract_environment import AbstractEnvironment


class FakeEnvironment(AbstractEnvironment):
    """A fake environment wraps models into an environment."""

    def __init__(
        self,
        dynamical_model,
        reward_model,
        initial_state_fn,
        termination_model=None,
        name=None,
    ):
        self.dynamical_model = dynamical_model
        self.reward_model = reward_model
        self.termination_model = termination_model
        self.initial_state_fn = initial_state_fn
        self._time = 0
        self._state = None
        self._name = name

    def reset(self):
        """Reset Simulation."""
        self._time = 0
        self._state = self.initial_state_fn()
        return self._state

    def step(self, action):
        """See `AbstractEnvironment.step'."""
        if self._state is None:
            raise AssertionError("Can't call step() before calling reset().")

        action = to_torch(action)
        next_state = self.next_state(self.state, action)
        reward = self.reward(self.state, action)
        done = self.done(self.state, action)
        info = self.info()
        return next_state, reward, done, info

    def next_state(self, state, action, next_state=None):
        """Calculate next state."""
        return sample_model(self.dynamical_model, state, action, next_state)

    def reward(self, state, action, next_state=None):
        """Calculate reward."""
        return sample_model(self.reward_model, state, action, next_state)

    def done(self, state, action, next_state=None):
        """Calculate termination."""
        if self.termination_model is None:
            return False
        return sample_model(self.termination_model, state, action, next_state)

    def info(self):
        """Get simulation information."""
        return {}

    @property
    def state(self):
        """See `AbstractEnvironment.state'."""
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def time(self):
        """See `AbstractEnvironment.time'."""
        return self._time

    @property
    def name(self):
        """Return class name."""
        return "Fake Environment" if self._name is None else self.name
