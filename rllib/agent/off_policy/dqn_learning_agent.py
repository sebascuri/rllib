"""Implementation of DQNAgent Algorithms."""
from rllib.algorithms.dqn import DQN

from .q_learning_agent import QLearningAgent


class DQNAgent(QLearningAgent):
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
    Mnih, Volodymyr, et al. (2015)
    Human-level control through deep reinforcement learning. Nature.
    """

    def __init__(self, q_function, policy, criterion, *args, **kwargs):
        super().__init__(
            q_function=q_function, policy=policy, criterion=criterion, *args, **kwargs
        )
        self.algorithm = DQN(
            critic=q_function, criterion=criterion(reduction="none"), gamma=self.gamma
        )
