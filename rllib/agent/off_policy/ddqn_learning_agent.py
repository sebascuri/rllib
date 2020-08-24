"""Implementation of DQNAgent Algorithms."""
from rllib.algorithms.ddqn import DDQN

from .q_learning_agent import QLearningAgent


class DDQNAgent(QLearningAgent):
    """Implementation of a DQN-Learning agent.

    Parameters
    ----------
    q_function: AbstractQFunction
        q_function that is learned.
    policy: QFunctionPolicy.
        Q-function derived policy.
    criterion: nn.Module
        Criterion to minimize the TD-error.

    References
    ----------
    Hasselt, H. V. (2010).
    Double Q-learning. NIPS.

    Van Hasselt, Hado, Arthur Guez, and David Silver. (2016)
    Deep reinforcement learning with double q-learning. AAAI.
    """

    def __init__(self, q_function, policy, criterion, *args, **kwargs):
        super().__init__(
            q_function=q_function, policy=policy, criterion=criterion, *args, **kwargs
        )
        self.algorithm = DDQN(
            critic=q_function, criterion=criterion(reduction="none"), gamma=self.gamma
        )
