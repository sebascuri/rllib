"""Two state Environment."""
from collections import defaultdict

from rllib.environment.mdp import MDP


class TwoStateProblem(MDP):
    """Implementation of Two State Problem.

    References
    ----------
    Bagnell, J. A., & Schneider, J. (2003).
    Covariant policy search. IJCAI.
    """

    def __init__(self):
        transitions = self._build_mdp()
        super().__init__(transitions, 2, 2)

    @staticmethod
    def _build_mdp():
        transitions = defaultdict(list)
        for state in range(2):
            for action in range(2):
                transitions[state, action].append(
                    {
                        "next_state": action,
                        "probability": 1.0,
                        "reward": state + 1 if state == action else 0,
                    }
                )
        return transitions
