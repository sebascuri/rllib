"""Interface for Grid World."""

from .mdp import MDP
import numpy as np

__all__ = ['EasyGridWorld']


class EasyGridWorld(MDP):
    """Easy implementation of a GridWorld from Sutton & Barto Example 3.5."""

    def __init__(self, width=5, height=5, num_actions=4, terminal_states=None):
        self.width = width
        self.height = height
        kernel, reward = self._build_mdp(num_actions, terminal_states)
        super().__init__(kernel, reward, terminal_states=terminal_states)

    def _build_mdp(self, num_actions, terminal_states=None):
        num_states = self.width * self.height
        kernel = np.zeros((num_states, num_actions, num_states))
        reward = np.zeros((num_states, num_actions))
        for state in range(num_states):
            for action in range(num_actions):
                g_state = self._state_to_grid(state)
                if (g_state == np.array([0, 1])).all():
                    g_next_state = np.array([self.height-1, 1])
                    r = 10
                elif (g_state == np.array([0, self.width-2])).all():
                    g_next_state = np.array([self.height // 2, self.width-2])
                    r = 5
                else:
                    g_action = self._action_to_grid(action, num_actions)
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

    @staticmethod
    def _action_to_grid(action, num_actions):
        if action >= num_actions:
            raise ValueError("action has to be < {}.".format(num_actions))

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
