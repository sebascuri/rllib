"""Random MDP environment."""
from collections import defaultdict

import numpy as np

from rllib.environment.mdp import MDP


class RandomMDP(MDP):
    """Implementation of a Random MDP.

    The MDP is sampled as P(s, a, s') ~ 1e-5 + U[0,1], r ~ U[0,1].

    Parameters
    ----------
    num_states: int
        Number of states in the mdp.
    num_actions: int.
        Number of actions in the mdp.

    References
    ----------
    Dann, C., Neumann, G., & Peters, J. (2014).
    Policy evaluation with temporal differences: A survey and comparison. JMLR.
    """

    def __init__(self, num_states=400, num_actions=10):
        transitions = self._build_mdp(num_states, num_actions)
        super().__init__(transitions, num_states, num_actions)

    @staticmethod
    def _build_mdp(num_states, num_actions):
        """Build the transition dictionary."""
        transitions = defaultdict(list)

        for state in range(num_states):
            for action in range(num_actions):
                p = np.random.rand(num_states) + 1e-5
                p = p / np.sum(p)
                r = np.random.rand()

                for next_state in range(num_states):
                    transitions[(state, action)].append(
                        {
                            "next_state": next_state,
                            "probability": p[next_state],
                            "reward": r,
                        }
                    )
        return transitions
