"""Interface for Markov Decision Processes."""

from collections import defaultdict
from itertools import product

import numpy as np
import torch
from gym import spaces

from rllib.environment.abstract_environment import AbstractEnvironment


class MDP(AbstractEnvironment):
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

    """

    def __init__(self, transitions, num_states, num_actions, initial_state=None,
                 terminal_states=None):

        observation_space = spaces.Discrete(num_states)
        action_space = spaces.Discrete(num_actions)
        super().__init__(dim_state=1, dim_action=1, dim_observation=1,
                         observation_space=observation_space, action_space=action_space,
                         num_states=num_states, num_actions=num_actions,
                         num_observations=num_states)

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
        next_state = []
        probs = []
        reward = 0
        for transition in transitions:
            next_state.append(transition['next_state'])
            probs.append(transition['probability'])
            reward += transition['reward'] * transition['probability']

        next_state = np.random.choice(next_state, p=probs)
        self.state = next_state

        if self.state in self.terminal_states:
            done = True
            next_state = self.num_states - 1
        else:
            done = False

        return next_state, reward, done, {}

    @staticmethod
    def check_transitions(transitions, num_states, num_actions):
        """Check if transitions belong to an MDP."""
        for state, action in product(range(num_states), range(num_actions)):
            p = 0
            for transition in transitions[(state, action)]:
                p += transition['probability']
                assert 'next_state' in transition
                assert 'reward' in transition

            np.testing.assert_allclose(p, 1.)


class TwoStateProblem(MDP):
    """Implementation of Two State Problem.

    References
    ----------
    Bagnell, J. A., & Schneider, J. (2003).
    Covariant policy search. IJCAI.
    """

    def __init__(self):

        transitions = defaultdict(list)
        for state in range(2):
            for action in range(2):
                transitions[state, action].append({
                    'next_state': action, 'probability': 1.,
                    'reward': action + 1 if state == action else 0})
        super().__init__(transitions, 2, 2)


class SingleChainProblem(MDP):
    """Implementation of Single Chain Problem.

    Parameters
    ----------
    chain_length: int
        number of states in chain.

    References
    ----------
    Furmston, T., & Barber, D. (2010).
    Variational methods for reinforcement learning. AISTATS
    """

    def __init__(self, chain_length=5):
        num_states = chain_length
        num_actions = 2
        transitions = defaultdict(list)

        for state in range(num_states - 1):
            transitions[(state, 0)].append(
                {'next_state': state + 1, 'probability': 1., 'reward': 0})

            transitions[(state, 1)].append(
                {'next_state': 0, 'probability': 1.,
                 'reward': 2})

        # Final transition.
        transitions[(num_states - 1, 1)].append(
            {'next_state': 0, 'probability': 1., 'reward': 2}
        )

        transitions[(num_states - 1, 0)].append(
            {'next_state': num_states - 1, 'probability': 1.,
             'reward': 2 * chain_length}
        )

        initial_state = 0
        super().__init__(transitions, num_states, num_actions, initial_state)


class DoubleChainProblem(MDP):
    """Implementation of Single Chain Problem.

    Parameters
    ----------
    chain_length: int
        number of states in chain.

    References
    ----------
    Furmston, T., & Barber, D. (2010).
    Variational methods for reinforcement learning. AISTATS
    """

    def __init__(self, chain_length=5):
        num_states = 2 * chain_length - 1
        num_actions = 2

        transitions = defaultdict(list)

        # Initial transition
        transitions[(0, 0)].append({'next_state': 1, 'probability': 1., 'reward': 0})
        transitions[(0, 1)].append({'next_state': chain_length, 'probability': 1.,
                                    'reward': 2})

        for i in range(chain_length - 2):
            top_state = 1 + i
            bottom_state = chain_length + i

            transitions[(top_state, 0)].append(
                {'next_state': 0, 'probability': 1., 'reward': 2}
            )
            transitions[(top_state, 1)].append(
                {'next_state': top_state + 1, 'probability': 1., 'reward': 0}
            )

            transitions[(bottom_state, 0)].append(
                {'next_state': 0, 'probability': 1., 'reward': 2}
            )
            transitions[(bottom_state, 1)].append(
                {'next_state': min(bottom_state + 1, num_states - 1), 'probability': 1.,
                 'reward': 0}
            )

        transitions[(chain_length - 1, 0)].append(
            {'next_state': 0, 'probability': 1., 'reward': 2}
        )

        transitions[(chain_length - 1, 1)].append(
            {'next_state': chain_length - 1, 'probability': 1.,
             'reward': 2 * chain_length}
        )

        transitions[(num_states - 1, 0)].append(
            {'next_state': 0, 'probability': 1., 'reward': 2}
        )

        transitions[(num_states - 1, 1)].append(
            {'next_state': num_states - 1, 'probability': 1., 'reward': chain_length}
        )

        initial_state = 0
        super().__init__(transitions, num_states, num_actions, initial_state)
