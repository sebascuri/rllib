"""Implementation of QLearning Algorithms."""
from rllib.agent.abstract_agent import AbstractAgent
from rllib.dataset import Observation
import torch
import numpy as np


class QLearningAgent(AbstractAgent):
    """Implementation of a Q-Learning agent.

    The Q-Learning algorithm implements the Q-Learning algorithm except for the
    computation of the TD-Error, which leads to different algorithms.

    Parameters
    ----------
    q_learning: QLearning
        Implementation of Q-Learning algorithm.
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

    Sutton, Richard S., et al. "Fast gradient-descent methods for temporal-difference
    learning with linear function approximation." Proceedings of the 26th Annual
    International Conference on Machine Learning. ACM, 2009.

    Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529-533.

    Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning
    with double q-learning." Thirtieth AAAI conference on artificial intelligence. 2016.

    """

    def __init__(self, q_learning, q_function, policy, criterion, optimizer, memory,
                 target_update_frequency=4, gamma=1.0,
                 exploration_steps=0, exploration_episodes=0):
        super().__init__(gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes)
        self.policy = policy
        self.q_learning = q_learning(q_function, criterion(reduction='none'),
                                     self.gamma)

        self.memory = memory
        self.target_update_frequency = target_update_frequency
        self.optimizer = optimizer

        self.logs['td_errors'] = []
        self.logs['episode_td_errors'] = []

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self.memory.append(observation)
        if self.memory.has_batch:
            self.train()
            if self.total_steps % self.target_update_frequency == 0:
                self.q_learning.update()

    def start_episode(self):
        """See `AbstractAgent.start_episode'."""
        super().start_episode()
        self.logs['episode_td_errors'].append([])

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        aux = self.logs['episode_td_errors'].pop(-1)
        if len(aux) > 0:
            self.logs['episode_td_errors'].append(np.abs(np.array(aux)).mean())

    def train(self, batches=1):
        """Train the DQN for `batches' batches.

        Parameters
        ----------
        batches: int

        """
        for batch in range(batches):
            observation, idx, weight = self.memory.get_batch()
            weight = torch.tensor(weight).float()
            observation = Observation(*map(lambda x: x.float(), observation))

            self.optimizer.zero_grad()
            ans = self.q_learning(
                observation.state, observation.action, observation.reward,
                observation.next_state, observation.done)

            loss = weight * ans.loss
            loss.mean().backward()

            self.optimizer.step()
            self.memory.update(idx, ans.td_error.numpy())

            self.logs['td_errors'].append(ans.td_error.mean().item())
            self.logs['episode_td_errors'][-1].append(ans.td_error.mean().item())
