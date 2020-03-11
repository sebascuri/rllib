"""Implementation of DQNAgent Algorithms."""
from rllib.agent import QLearningAgent
from rllib.algorithms.q_learning import SoftQLearning


class SoftQLearningAgent(QLearningAgent):
    """Implementation of a DQN-Learning agent.

    Parameters
    ----------
    q_function: AbstractQFunction
        q_function that is learned.
    criterion: nn.Module
        Criterion to minimize the TD-error.
    optimizer: nn.optim
        Optimization algorithm for q_function.
    memory: ExperienceReplay
        Memory where to store the observations.
    temperature: ParameterDecay.
        Temperature of Soft Q function.
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

    def __init__(self, q_function, criterion, optimizer,
                 memory, temperature, target_update_frequency=4, gamma=1.0,
                 exploration_steps=0, exploration_episodes=0):
        super().__init__(q_function, None, criterion, optimizer, memory,
                         target_update_frequency, gamma,
                         exploration_steps, exploration_episodes)
        self.q_learning = SoftQLearning(q_function, criterion(reduction='none'),
                                        temperature, self.gamma)
        self.policy = self.q_learning.policy
