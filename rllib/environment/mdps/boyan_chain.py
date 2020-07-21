"""Boyan Chain Environment."""
from collections import defaultdict

from rllib.environment.mdp import MDP


class BoyanChain(MDP):
    """Boyan Chain environment is common for off-policy PE.

    The environment consists of `num_states' states and one action.

    Parameters
    ----------
    num_states: int, optional (default=7).
        Number of states in the MDP.

    References
    ----------
    Boyan, J. A. (2002).
    Technical update: Least-squares temporal difference learning.
    Machine learning, 49(2-3), 233-246.
    """

    def __init__(self, num_states=7):
        transitions = self._build_mdp(num_states)

        super().__init__(
            transitions=transitions,
            num_states=num_states,
            num_actions=1,
            initial_state=0,
            terminal_states=None,
        )

    @staticmethod
    def _build_mdp(num_states):
        """Build the transition dictionary."""
        transitions = defaultdict(list)
        for state in range(num_states - 2):
            transitions[(state, 0)].append(
                {"next_state": state + 1, "reward": -3, "probability": 0.5}
            )
            transitions[(state, 0)].append(
                {"next_state": state + 2, "reward": -3, "probability": 0.5}
            )

        transitions[(num_states - 2, 0)].append(
            {"next_state": num_states - 1, "reward": -2, "probability": 1.0}
        )
        transitions[(num_states - 1, 0)].append(
            {"next_state": num_states - 1, "reward": 0, "probability": 1.0}
        )

        return transitions
