"""Implementation of DQNAgent Algorithms."""
import torch.nn.modules.loss as loss

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

    def __init__(self, critic, policy, criterion=loss.MSELoss, *args, **kwargs):
        super().__init__(critic=critic, policy=policy, *args, **kwargs)
        self.algorithm = DDQN(
            policy=policy,
            critic=critic,
            criterion=criterion(reduction="none"),
            *args,
            **kwargs,
        )
