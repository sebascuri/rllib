"""Implementation of DQNAgent Algorithms."""
from rllib.algorithms.dqn import DQN

from .q_learning_agent import QLearningAgent


class DQNAgent(QLearningAgent):
    """Implementation of a DQN-Learning agent.

    Parameters
    ----------
    critic: AbstractQFunction
        critic that is learned.
    policy: QFunctionPolicy.
        Q-function derived policy.
    criterion: nn.Module
        Criterion to minimize the TD-error.

    References
    ----------
    Mnih, Volodymyr, et al. (2015)
    Human-level control through deep reinforcement learning. Nature.
    """

    def __init__(self, critic, policy, *args, **kwargs):
        super().__init__(critic=critic, policy=policy, *args, **kwargs)
        self.algorithm = DQN(
            policy=policy,
            critic=critic,
            criterion=self.algorithm.criterion,
            gamma=self.gamma,
        )
