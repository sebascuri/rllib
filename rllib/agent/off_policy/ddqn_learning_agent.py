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
    optimizer: nn.optim
        Optimization algorithm for q_function.
    memory: ExperienceReplay
        Memory where to store the observations.
    target_update_frequency: int
        How often to update the q_function target.
    gamma: float, optional
        Discount factor.
    exploration_steps: int, optional
        Number of random exploration steps.
    exploration_episodes: int, optional
        Number of random exploration steps.

    References
    ----------
    Hasselt, H. V. (2010).
    Double Q-learning. NIPS.

    Van Hasselt, Hado, Arthur Guez, and David Silver. (2016)
    Deep reinforcement learning with double q-learning. AAAI.
    """

    def __init__(
        self, q_function, policy, criterion, optimizer, memory, *args, **kwargs
    ):
        super().__init__(
            q_function=q_function,
            policy=policy,
            criterion=criterion,
            optimizer=optimizer,
            memory=memory,
            *args,
            **kwargs,
        )
        self.algorithm = DDQN(q_function, criterion(reduction="none"), self.gamma)
