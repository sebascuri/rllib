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
    Mnih, Volodymyr, et al. (2015)
    Human-level control through deep reinforcement learning. Nature.
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
        self.algorithm = DQN(
            critic=q_function, criterion=criterion(reduction="none"), gamma=self.gamma
        )
