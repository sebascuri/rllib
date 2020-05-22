"""Implementation of QLearning Algorithms."""

from .off_policy_agent import OffPolicyAgent
from rllib.algorithms.q_learning import QLearning


class QLearningAgent(OffPolicyAgent):
    """Implementation of a Q-Learning agent.

    The Q-Learning algorithm implements the Q-Learning algorithm except for the
    computation of the TD-Error, which leads to different algorithms.

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
    Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.

    Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529-533.

    Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning
    with double q-learning." Thirtieth AAAI conference on artificial intelligence. 2016.

    """

    def __init__(self, env_name, q_function, policy, criterion, optimizer,
                 memory, num_iter=1, batch_size=64, target_update_frequency=4,
                 train_frequency=1, num_rollouts=0,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0, comment=''):
        super().__init__(env_name, optimizer=optimizer, memory=memory,
                         batch_size=batch_size,
                         train_frequency=train_frequency, num_rollouts=num_rollouts,
                         num_iter=num_iter,
                         target_update_frequency=target_update_frequency,
                         gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes, comment=comment)
        self.policy = policy
        self.algorithm = QLearning(q_function, criterion(reduction='none'), gamma)
