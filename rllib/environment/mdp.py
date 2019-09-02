"""Interface for Markov Decision Processes."""


from rllib.environment.abstract_environment import AbstractEnvironment
from gym import spaces
import numpy as np
import torch
from torch.distributions import Categorical


__all__ = ['MDP', 'EasyGridWorld']


class MDP(AbstractEnvironment):
    """Interface MDP environments.

    Parameters
    ----------
    transition_kernel: nd-array [num_states x num_actions x num_states]
    reward: nd-array [num_states x num_actions]
    terminal_states: list of int, optional
    initial_state: callable or int, optional
    max_steps: max_number_of_steps, optional

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
                 terminal_states: list = None, max_steps=None):

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
        self._max_steps = max_steps if max_steps else float('Inf')

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
        self.state = next_state.sample().item()

        if self.state in self.terminal_states or self._time >= self._max_steps:
            done = True
        else:
            done = False

        return self.state, reward, done, {}

    @property
    def state(self):
        """See `AbstractEnvironment.state'."""
        return self._state

    def reset(self):
        """See `AbstractEnvironment.reset'."""
        self._state = self.initial_state()
        self._time = 0
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def time(self):
        """See `AbstractEnvironment.time'."""
        return self._time


class EasyGridWorld(MDP):
    """Easy implementation of a GridWorld from Sutton & Barto Example 3.5."""

    def __init__(self, width=5, height=5, num_actions=4, terminal_states: list = None,
                 max_steps=None):
        self.width = width
        self.height = height
        self.num_states = width * height
        self.num_actions = num_actions
        super().__init__(*self._build_mdp(terminal_states),
                         terminal_states=terminal_states, max_steps=max_steps)

    def _build_mdp(self, terminal_states: list = None):
        kernel = np.zeros((self.num_states, self.num_actions, self.num_states))
        reward = np.zeros((self.num_states, self.num_actions))
        for state in range(self.num_states):
            for action in range(self.num_actions):
                g_state = self._state_to_grid(state)
                if (g_state == np.array([0, 1])).all():
                    g_next_state = np.array([self.height-1, 1])
                    r = 10
                elif (g_state == np.array([0, self.width-2])).all():
                    g_next_state = np.array([self.height // 2, self.width-2])
                    r = 5
                else:
                    g_action = self._action_to_grid(action)
                    g_next_state = g_state + g_action
                    if not self._is_valid(g_next_state):
                        g_next_state = g_state
                        r = -1
                    else:
                        r = 0

                next_state = self._grid_to_state(g_next_state)
                if state in terminal_states if terminal_states else []:
                    kernel[state, action, state] = 1
                    reward[state, action] = 0
                else:
                    kernel[state, action, next_state] = 1
                    reward[state, action] = r

        return kernel, reward

    def _state_to_grid(self, state):
        return np.array([state // self.width, state % self.width])

    def _grid_to_state(self, grid_state):
        return grid_state[0] * self.width + grid_state[1]

    def _action_to_grid(self, action):
        if action >= self.num_actions:
            raise ValueError("action has to be < {}.".format(self.num_actions))

        if action == 0:  # Down
            return np.array([1, 0])
        elif action == 1:  # Up
            return np.array([-1, 0])
        elif action == 2:  # Right
            return np.array([0, 1])
        elif action == 3:  # Left
            return np.array([0, -1])
        elif action == 4:  # Down Right
            return np.array([1, 1])
        elif action == 5:  # Up Right
            return np.array([-1, 1])
        elif action == 6:  # Down Left
            return np.array([1, -1])
        elif action == 7:  # Up Left
            return np.array([-1, -1])

    def _is_valid(self, grid_state):
        return 0 <= grid_state[0] < self.height and 0 <= grid_state[1] < self.width
