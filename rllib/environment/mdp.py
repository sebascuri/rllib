"""Interface for Markov Decision Processes."""

from itertools import product
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from gym import Env, spaces

from rllib.environment.abstract_environment import AbstractEnvironment

Transition = Dict[Tuple[int, int], List[Dict[str, Union[float, int]]]]


class MDP(AbstractEnvironment, Env):
    """Interface MDP environments.

    Parameters
    ----------
    transitions: dict.
        Mapping from (state, action) tuples to a list of transitions.
        A transition is a dictionary with keys 'reward', 'next_state' and 'probability'.
    num_states: int.
    num_actions: int.
    terminal_states: list of int, optional
        For each state, action
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

    TODO: Add non-sparse MDPs (such as random MDPs).
    """

    def __init__(
        self,
        transitions,
        num_states,
        num_actions,
        initial_state=None,
        terminal_states=None,
    ):

        observation_space = spaces.Discrete(num_states)
        action_space = spaces.Discrete(num_actions)
        super().__init__(
            dim_state=(),
            dim_action=(),
            dim_observation=(),
            observation_space=observation_space,
            action_space=action_space,
            num_states=num_states,
            num_actions=num_actions,
            num_observations=num_states,
        )

        if initial_state is None:
            self.initial_state = lambda: self.observation_space.sample()
        elif not callable(initial_state):
            self.initial_state = lambda: initial_state
        else:
            self.initial_state = initial_state

        self._state = self.initial_state()
        self._time = 0

        self.check_transitions(transitions, num_states, num_actions)
        self.transitions = transitions

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
        if isinstance(action, torch.Tensor):
            action = action.item()
        transitions = self.transitions[(self.state, action)]
        probs = []
        for transition in transitions:
            probs.append(transition["probability"])

        next_transition_idx = np.random.choice(len(probs), p=probs)
        self.state = transitions[next_transition_idx]["next_state"]
        reward = transitions[next_transition_idx]["reward"]

        if self.state in self.terminal_states:
            done = True
            next_state = self.num_states - 1
        else:
            done = False
            next_state = self.state

        return next_state, reward, done, {}

    @staticmethod
    def check_transitions(transitions, num_states, num_actions):
        """Check if transitions belong to an MDP."""
        for state, action in product(range(num_states), range(num_actions)):
            p = 0
            for transition in transitions[(state, action)]:
                p += transition["probability"]
                assert "next_state" in transition
                assert "reward" in transition

            np.testing.assert_allclose(p, 1.0)

        states, actions = set(), set()
        for (state, action) in transitions:
            states.add(state), actions.add(action)
        assert len(states) == num_states
        assert len(actions) == num_actions
