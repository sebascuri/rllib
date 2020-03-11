"""Implementation of QLearning Algorithms."""
from rllib.agent.abstract_agent import AbstractAgent
from rllib.algorithms.q_learning import QLearning
from rllib.dataset import Observation
from rllib.util.logger import Logger
import torch


class QLearningAgent(AbstractAgent):
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

    def __init__(self, q_function, policy, criterion, optimizer,
                 memory, target_update_frequency=4, gamma=1.0,
                 exploration_steps=0, exploration_episodes=0):
        super().__init__(gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes)
        self.policy = policy
        self.q_learning = QLearning(q_function, criterion(reduction='none'), self.gamma)

        self.memory = memory
        self.target_update_frequency = target_update_frequency
        self.optimizer = optimizer

        self.logs['td_errors'] = Logger('abs_mean')
        self.logs['losses'] = Logger('mean')

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self.memory.append(observation)
        if self.memory.has_batch:
            if self._training:
                self._train()
            if self.total_steps % self.target_update_frequency == 0:
                self.q_learning.update()

    def _train(self):
        """Train the Q-Learning Agent."""
        observation, idx, weight = self.memory.get_batch()
        weight = torch.tensor(weight).float()
        observation = Observation(*map(lambda x: x.float(), observation))

        self.optimizer.zero_grad()
        losses = self.q_learning(
            observation.state, observation.action, observation.reward,
            observation.next_state, observation.done)

        loss = (weight * losses.loss).mean()
        loss.backward()

        self.optimizer.step()
        self.memory.update(idx, losses.td_error.numpy())

        self.logs['td_errors'].append(losses.td_error.mean().item())
        self.logs['losses'].append(loss.item())
