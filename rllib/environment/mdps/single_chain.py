"""Single Chain Environment."""
from collections import defaultdict

from rllib.environment.mdp import MDP


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
        transitions = self._build_mdp(num_states)
        initial_state = 0
        super().__init__(transitions, num_states, num_actions, initial_state)

    @staticmethod
    def _build_mdp(chain_length):
        """Build the transition dictionary."""
        transitions = defaultdict(list)
        num_states = chain_length

        for state in range(num_states - 1):
            transitions[(state, 0)].append(
                {"next_state": state + 1, "probability": 1.0, "reward": 0}
            )

            transitions[(state, 1)].append(
                {"next_state": 0, "probability": 1.0, "reward": 2}
            )

        # Final transition.
        transitions[(num_states - 1, 1)].append(
            {"next_state": 0, "probability": 1.0, "reward": 2}
        )

        transitions[(num_states - 1, 0)].append(
            {
                "next_state": num_states - 1,
                "probability": 1.0,
                "reward": 2 * chain_length,
            }
        )

        return transitions
