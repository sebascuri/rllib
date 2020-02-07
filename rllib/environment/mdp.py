"""Interface for Markov Decision Processes."""


from rllib.environment.abstract_environment import AbstractEnvironment
from gym import spaces
import torch
from torch.distributions import Categorical


__all__ = ['MDP']


class MDP(AbstractEnvironment):
    """Interface MDP environments.

    Parameters
    ----------
    transition_kernel: nd-array [num_states x num_actions x num_states]
    reward: nd-array [num_states x num_actions]
    terminal_states: list of int, optional
    initial_state: callable or int, optional

    Methods
    -------
    state: int
        return the current state of the system.
    time: int
        return the current time step.
    reset(state):
        reset the state.
    step(action): int
        execute a one step simulation and return the next state.

    """

    def __init__(self, transition_kernel, reward, initial_state=None,
                 terminal_states=None):

        self.num_states = transition_kernel.shape[0]
        self.num_actions = transition_kernel.shape[1]

        observation_space = spaces.Discrete(self.num_states)
        action_space = spaces.Discrete(self.num_actions)
        super().__init__(dim_state=1, dim_action=1, dim_observation=1,
                         observation_space=observation_space, action_space=action_space,
                         num_states=self.num_states, num_actions=self.num_actions,
                         num_observations=self.num_states)

        if initial_state is None:
            self.initial_state = lambda: self.observation_space.sample()
        elif not callable(initial_state):
            self.initial_state = lambda: initial_state
        else:
            self.initial_state = initial_state

        self._state = self.initial_state()
        self._time = 0
        self.kernel = transition_kernel
        self.reward = reward
        self.terminal_states = terminal_states if terminal_states is not None else []

    @property
    def state(self):
        """Return the state of the system."""
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    def reset(self):
        """Reset MDP environment."""
        self._state = self.initial_state()
        self._time = 0
        return self._state

    @property
    def time(self):
        """Return the current time of the system."""
        return self._time

    def step(self, action):
        """Do a one step ahead simulation of the system.

        s_{t+1} ~ P(s_t, a_t)
        r = R(s_t, a_t, s_{t+1})

        Parameters
        ----------
        action: int

        Returns
        -------
        next_state: int
        reward: float
        done: bool
        info: dict

        """
        self._time += 1
        next_state = Categorical(torch.tensor(self.kernel[self.state, action]))
        reward = self.reward[self.state, action]
        next_state = next_state.sample().item()
        self.state = next_state

        if self.state in self.terminal_states:
            done = True
            next_state = self.num_states - 1
        else:
            done = False

        return next_state, reward, done, {}
