"""Double Chain Environment."""
from collections import defaultdict

from rllib.environment.mdp import MDP


class DoubleChainProblem(MDP):
    """Implementation of Double Chain Problem.

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
        transitions = self._build_mdp(chain_length)
        initial_state = 0
        super().__init__(transitions, num_states, num_actions, initial_state)

    @staticmethod
    def _build_mdp(chain_length):
        """Build the transition dictionary."""
        transitions = defaultdict(list)

        num_states = 2 * chain_length - 1

        # Initial transition
        transitions[(0, 0)].append({"next_state": 1, "probability": 1.0, "reward": 0})
        transitions[(0, 1)].append(
            {"next_state": chain_length, "probability": 1.0, "reward": 2}
        )

        for i in range(chain_length - 2):
            top_state = 1 + i
            bottom_state = chain_length + i

            transitions[(top_state, 1)].append(
                {"next_state": 0, "probability": 1.0, "reward": 2}
            )
            transitions[(top_state, 0)].append(
                {"next_state": top_state + 1, "probability": 1.0, "reward": 0}
            )

            transitions[(bottom_state, 1)].append(
                {"next_state": 0, "probability": 1.0, "reward": 2}
            )
            transitions[(bottom_state, 0)].append(
                {
                    "next_state": min(bottom_state + 1, num_states - 1),
                    "probability": 1.0,
                    "reward": 0,
                }
            )

        transitions[(chain_length - 1, 1)].append(
            {"next_state": 0, "probability": 1.0, "reward": 2}
        )

        transitions[(chain_length - 1, 0)].append(
            {
                "next_state": chain_length - 1,
                "probability": 1.0,
                "reward": 2 * chain_length,
            }
        )

        transitions[(num_states - 1, 1)].append(
            {"next_state": 0, "probability": 1.0, "reward": 2}
        )

        transitions[(num_states - 1, 0)].append(
            {"next_state": num_states - 1, "probability": 1.0, "reward": chain_length}
        )

        return transitions
