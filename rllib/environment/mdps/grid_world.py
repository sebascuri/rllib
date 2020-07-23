"""Interface for Grid World."""

from collections import defaultdict

import numpy as np

from rllib.environment.mdp import MDP


class EasyGridWorld(MDP):
    """Easy implementation of a GridWorld.

    There is a transition from A->A' [0, 1] -> [height-1, 1] with reward 10.
    There is a transition from B->B' [0, width-2] -> [height//2, width-2] with reward 5.

    When noise > 0, then transitions happen with probability = 1-noise.
    With probability noise the agent stays at the current state.

    References
    ----------
    Sutton, R. S., & Barto, A. G. (2018).
    Reinforcement learning: An introduction. MIT press.
    Example 3.2 p. 60.
    """

    def __init__(self, width=5, height=5, num_actions=4, noise=0, terminal_states=None):
        self.width = width
        self.height = height
        transitions = self._build_mdp(num_actions, noise, terminal_states)
        super().__init__(
            transitions, width * height, num_actions, terminal_states=terminal_states
        )

    def _build_mdp(self, num_actions, noise, terminal_states=None):
        """Build transition dictionary."""
        transitions = defaultdict(list)
        num_states = self.width * self.height
        for state in range(num_states):
            for action in range(num_actions):
                g_state = self._state_to_grid(state)
                if (g_state == np.array([0, 1])).all():
                    g_next_state = np.array([self.height - 1, 1])
                    r = 10
                elif (g_state == np.array([0, self.width - 2])).all():
                    g_next_state = np.array([self.height // 2, self.width - 2])
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
                    transitions[(state, action)].append(
                        {"next_state": state, "reward": 0, "probability": 1.0}
                    )
                elif r > 0:
                    transitions[(state, action)].append(
                        {"next_state": next_state, "reward": r, "probability": 1.0}
                    )
                else:
                    transitions[(state, action)].append(
                        {
                            "next_state": next_state,
                            "reward": r,
                            "probability": 1.0 - noise,
                        }
                    )
                    transitions[(state, action)].append(
                        {"next_state": state, "reward": 0, "probability": noise}
                    )

        return transitions

    def _state_to_grid(self, state):
        """Transform a state in [0, N-1] to a grid position."""
        return np.array([state // self.width, state % self.width])

    def _grid_to_state(self, grid_state):
        """Transform a grid position to a state in [0, N-1]."""
        return grid_state[0] * self.width + grid_state[1]

    @staticmethod
    def _action_to_grid(action):
        """Transform an action to a grid action."""
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
        """Check if the grid state is valid."""
        return 0 <= grid_state[0] < self.height and 0 <= grid_state[1] < self.width
