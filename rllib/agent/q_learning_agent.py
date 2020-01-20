"""Implementation of QLearning Algorithms."""
from .abstract_agent import AbstractAgent
from abc import abstractmethod
import torch
from torch.distributions import Categorical
import numpy as np
import copy

__all__ = ['QLearningAgent', 'GQLearningAgent', 'DQNAgent', 'DDQNAgent']


class AbstractQLearningAgent(AbstractAgent):
    """Abstract Implementation of the Q-Learning Algorithm.

    The AbstractQLearning algorithm implements the Q-Learning algorithm except for the
    computation of the TD-Error, which leads to different algorithms.

    Parameters
    ----------
    q_function: AbstractQFunction
        q_function that is learned.
    exploration: AbstractExplorationStrategy.
        exploration strategy that returns the actions.
    criterion: nn.Module
    optimizer: nn.optim
    memory: ExperienceReplay
        memory where to store the observations.

    References
    ----------
    Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.

    """

    def __init__(self, q_function, exploration, criterion, optimizer, memory,
                 target_update_frequency=4, gamma=1.0, episode_length=None):
        super().__init__(gamma=gamma, episode_length=episode_length)
        self._q_function = q_function
        self._q_target = copy.deepcopy(q_function)
        self._exploration = exploration
        self._criterion = criterion
        self._memory = memory
        self._target_update_frequency = target_update_frequency
        self._optimizer = optimizer

        # self.logs['q_function'] = []
        self.logs['td_errors'] = []
        self.logs['episode_td_errors'] = []

    def act(self, state):
        """See `AbstractAgent.act'."""
        logits = self._q_function(torch.tensor(state).float())
        action_distribution = Categorical(logits=logits)
        if self.training:
            action = self._exploration(action_distribution, self.total_steps).item()
        else:
            action = torch.argmax(action_distribution.logits).item()

        return action

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self._memory.append(observation)
        if self._memory.has_batch:
            self._train()
            if self.total_steps % self._target_update_frequency == 0:
                self._q_target.parameters = self._q_function.parameters

    def start_episode(self):
        """See `AbstractAgent.start_episode'."""
        super().start_episode()
        self.logs['episode_td_errors'].append([])

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        aux = self.logs['episode_td_errors'].pop(-1)
        if len(aux) > 0:
            self.logs['episode_td_errors'].append(np.abs(np.array(aux)).mean())

    @property
    def policy(self):
        """See `AbstractAgent.policy'."""
        return self._q_function.extract_policy(temperature=0.001)

    def _train(self, batches=1):
        """Train the DQN for `batches' batches.

        Parameters
        ----------
        batches: int

        """
        for batch in range(batches):
            (state, action, reward, next_state, done), idx, w = self._memory.get_batch()
            self._optimizer.zero_grad()
            pred_q, target_q = self._td(state.float(), action.float(), reward.float(),
                                        next_state.float(), done.float())

            td_error = (pred_q.detach() - target_q.detach()).mean().item()
            self.logs['td_errors'].append(td_error)
            self.logs['episode_td_errors'][-1].append(td_error)
            loss = self._criterion(pred_q, target_q, reduction='none')
            loss = torch.tensor(w).float() * loss
            loss.mean().backward()
            self._optimizer.step()

    @abstractmethod
    def _td(self, state, action, reward, next_state, done):
        raise NotImplementedError


class QLearningAgent(AbstractQLearningAgent):
    """Implementation of Q-Learning algorithm.

    loss = l[Q(x, a), r + Q(x', arg max Q(x', a))]

    References
    ----------
    Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.
    """

    def _td(self, state, action, reward, next_state, done):
        pred_q = self._q_function(state, action)

        # target = r + gamma * max Q(x', a) and don't stop gradient.
        target_q = self._q_function.max(next_state)
        target_q = reward + self.gamma * target_q * (1 - done)

        return pred_q, target_q


class GQLearningAgent(AbstractQLearningAgent):
    """Implementation of Gradient Q-Learning algorithm.

    loss = l[Q(x, a), r + Q(x', arg max Q(x', a)).stop_gradient]

    References
    ----------
    Sutton, Richard S., et al. "Fast gradient-descent methods for temporal-difference
    learning with linear function approximation." Proceedings of the 26th Annual
    International Conference on Machine Learning. ACM, 2009.

    """

    def _td(self, state, action, reward, next_state, done):
        pred_q = self._q_function(state, action)

        # target = r + gamma * max Q(x', a) and stop gradient.
        next_q = self._q_function.max(next_state)
        target_q = reward + self.gamma * next_q * (1 - done)

        return pred_q, target_q.detach()


class DQNAgent(AbstractQLearningAgent):
    """Implementation of DQN algorithm.

    loss = l[Q(x, a), r + max_a Q'(x', a)]

    References
    ----------
    Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529.
    """

    def _td(self, state, action, reward, next_state, done):
        pred_q = self._q_function(state, action)

        # target = r + gamma * max Q_target(x', a)
        next_q = self._q_target.max(next_state)
        target_q = reward + self.gamma * next_q * (1 - done)

        return pred_q, target_q.detach()


class DDQNAgent(AbstractQLearningAgent):
    """Implementation of Double DQN algorithm.

    loss = l[Q(x, a), r + Q'(x', argmax Q(x,a))]

    References
    ----------
    Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning
    with double q-learning." Thirtieth AAAI conference on artificial intelligence. 2016.
    """

    def _td(self, state, action, reward, next_state, done):
        pred_q = self._q_function(state, action)

        # target = r + gamma * Q_target(x', argmax Q(x', a))

        next_action = self._q_function.argmax(next_state)
        next_q = self._q_target(next_state, next_action)
        target_q = reward + self.gamma * next_q * (1 - done)

        return pred_q, target_q.detach()
