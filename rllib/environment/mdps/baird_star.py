"""Baird Star Environment."""
from collections import defaultdict

import numpy as np

from rllib.environment.mdp import MDP


class BairdStar(MDP):
    """Baird star environment is common for off-policy PE.

    The environment consists of 7 states and 2 actions.

    When the zero action is selected, the next state is 0 (star center).
    When the one action is selected, the next state is one of the edge states at random.

    Parameters
    ----------
    num_states: int, optional (default=7).
        Number of states in the MDP.

    References
    ----------
    Baird, L. (1995).
    Residual algorithms: Reinforcement learning with function approximation.
    In Machine Learning Proceedings.
    """

    def __init__(self, num_states=7):
        transitions = self._build_mdp(num_states)
        self.feature_matrix = self._build_feature_matrix(num_states)

        super().__init__(
            transitions=transitions,
            num_states=num_states,
            num_actions=2,
            initial_state=None,
            terminal_states=None,
        )

    @staticmethod
    def _build_mdp(num_states):
        """Build the transition dictionary."""
        transitions = defaultdict(list)
        for state in range(num_states):
            transitions[(state, 0)].append(
                {"next_state": 0, "reward": 0, "probability": 1.0}
            )
            for next_state in range(1, num_states):
                transitions[(state, 1)].append(
                    {
                        "next_state": next_state,
                        "reward": 0,
                        "probability": 1 / (num_states - 1),
                    }
                )
        return transitions

    @staticmethod
    def _build_feature_matrix(num_states):
        """Build the feature matrix of the environment."""
        phi = 2 * np.eye(num_states)
        phi[0, 0] = 1
        phi[:, -1] = 1
        phi[0, -1] = 2
        return phi
