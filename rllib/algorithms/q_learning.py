"""Q Learning Algorithm."""

import torch.nn as nn
import copy
from collections import namedtuple

QLearningLoss = namedtuple('QLearningLoss', ['loss', 'td_error'])


class QLearning(nn.Module):
    r"""Implementation of Q-Learning algorithm.

    The Q-Learning algorithm attempts to find the fixed point of:
    .. math:: Q(s, a) = r(s, a) + \gamma \max_a Q(s', a)

    Usually the loss is computed as:
    .. math:: Q_{target} = r(s, a) + \gamma \max_a Q(s', a)
    .. math:: \mathcal{L}(Q(s, a), Q_{target})

    Parameters
    ----------
    q_function: AbstractQFunction
        q_function to optimize.
    criterion: _Loss
        Criterion to optimize.
    gamma: float
        discount factor.

    References
    ----------
    Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.
    """

    def __init__(self, q_function, criterion, gamma):
        super().__init__()
        self.q_function = q_function
        self.q_target = copy.deepcopy(q_function)
        self.criterion = criterion
        self.gamma = gamma

    def forward(self, state, action, reward, next_state, done):
        """Compute the loss and the td-error."""
        pred_q = self.q_function(state, action)

        # target = r + gamma * max Q(x', a) and don't stop gradient.
        target_q = self.q_function(next_state).max(dim=-1)[0]
        target_q = reward + self.gamma * target_q * (1 - done)

        return self._build_return(pred_q, target_q)

    def _build_return(self, pred_q, target_q):
        return QLearningLoss(loss=self.criterion(pred_q, target_q),
                             td_error=(pred_q - target_q).detach())

    def update(self):
        """Update the target network."""
        self.q_target.update_parameters(self.q_function.parameters())


class SemiGQLearning(QLearning):
    r"""Implementation of Semi-Gradient Q Learning algorithm.

    The semi-gradient q-learning algorithm detaches the gradient of the target value.

    .. math:: Q_{target} = (r(s, a) + \gamma \max_a Q(s', a)).detach()

    References
    ----------
    Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.

    Sutton, Richard S., et al. "Fast gradient-descent methods for temporal-difference
    learning with linear function approximation." Proceedings of the 26th Annual
    International Conference on Machine Learning. ACM, 2009.
    """

    def forward(self, state, action, reward, next_state, done):
        """Compute the loss and the td-error."""
        pred_q = self.q_function(state, action)

        # target = r + gamma * max Q(x', a) and stop gradient.
        target_q = self.q_function(next_state).max(dim=-1)[0]
        target_q = reward + self.gamma * target_q * (1 - done)

        return self._build_return(pred_q, target_q.detach())


class DQN(QLearning):
    r"""Implementation of Deep Q Learning algorithm.

    The deep q-learning algorithm has a separate target network for the target value.

    Q_{target} = (r(s, a) + \gamma \max_a Q_{target}(s', a)).detach()

    References
    ----------
    Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.

    Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529-533.
    """

    def forward(self, state, action, reward, next_state, done):
        """Compute the loss and the td-error."""
        pred_q = self.q_function(state, action)

        # target = r + gamma * max Q(x', a) and stop gradient.
        target_q = self.q_target(next_state).max(dim=-1)[0]
        target_q = reward + self.gamma * target_q * (1 - done)

        return self._build_return(pred_q, target_q.detach())


class DDQN(QLearning):
    r"""Implementation of Double Q Learning algorithm.

    The double q-learning algorithm calculates the target value with the action that
    maximizes the primal function to mitigate over-estimation bias.

    a_{target} = \arg max_a Q(s', a)
    Q_{target} = (r(s, a) + \gamma \max_a Q_{target}(s', a_{target})).detach()

    References
    ----------
    Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.

    Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning
    with double q-learning." Thirtieth AAAI conference on artificial intelligence. 2016.
    """

    def forward(self, state, action, reward, next_state, done):
        """Compute the loss and the td-error."""
        pred_q = self.q_function(state, action)

        next_action = self.q_function(next_state).argmax(dim=-1)
        next_q = self.q_target(next_state, next_action)
        target_q = reward + self.gamma * next_q * (1 - done)

        return self._build_return(pred_q, target_q.detach())
