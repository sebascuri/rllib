"""SARSA Algorithm."""

import torch.nn as nn
import copy
from .q_learning import QLearningLoss


class SARSA(nn.Module):
    r"""Implementation of SARSA algorithm.

    SARSA is an on-policy model-free control algorithm.

    The SARSA algorithm attempts to find the fixed point of:
    .. math:: Q(s, a) = r(s, a) + \gamma Q(s', a')
    where a' is sampled from a greedy policy w.r.t the current Q-Value estimate.

    Usually the loss is computed as:
    .. math:: Q_{target} = r(s, a) + \gamma Q(s', a')
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
    TODO: Find.
    """

    def __init__(self, q_function, criterion, gamma):
        super().__init__()
        self.q_function = q_function
        self.q_target = copy.deepcopy(q_function)
        self.criterion = criterion
        self.gamma = gamma

    def forward(self, state, action, reward, next_state, done, next_action, policy):
        """Compute the loss and the td-error."""
        pred_q = self.q_function(state, action)

        next_v = self.q_function(next_state, next_action)
        target_q = reward + self.gamma * next_v * (1 - done)

        return self._build_return(pred_q, target_q)

    def _build_return(self, pred_q, target_q):
        return QLearningLoss(loss=self.criterion(pred_q, target_q),
                             td_error=(pred_q - target_q).detach())

    def update(self):
        """Update the target network."""
        self.q_target.update_parameters(self.q_function.parameters())


class SemiGSARSA(SARSA):
    r"""Implementation of Semi-Gradient SARSA.

    The semi-gradient SARSA algorithm detaches the gradient of the target value.

    .. math:: Q_{target} = (r(s, a) + \gamma Q(s', a')).detach()

    References
    ----------
    TODO: find

    Sutton, Richard S., et al. "Fast gradient-descent methods for temporal-difference
    learning with linear function approximation." Proceedings of the 26th Annual
    International Conference on Machine Learning. ACM, 2009.
    """

    def forward(self, state, action, reward, next_state, done, next_action, policy):
        """Compute the loss and the td-error."""
        pred_q = self.q_function(state, action)

        next_v = self.q_function(next_state, next_action)
        target_q = reward + self.gamma * next_v * (1 - done)

        return self._build_return(pred_q, target_q.detach())


class DSARSA(SARSA):
    r"""Implementation of Delayed SARSA algorithm.

    The delayed SARSA algorithm has a separate target network for the target value.

    .. math:: Q_{target} = (r(s, a) + \gamma  Q_{target}(s', a')).detach()

    References
    ----------
    TODO: find

    Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529-533.
    """

    def forward(self, state, action, reward, next_state, done, next_action, policy):
        """Compute the loss and the td-error."""
        pred_q = self.q_function(state, action)

        next_v = self.q_target(next_state, next_action)
        target_q = reward + self.gamma * next_v * (1 - done)

        return self._build_return(pred_q, target_q.detach())


class ExpectedSARSA(SARSA):
    r"""Implementation of Expected SARSA algorithm.

    The expected SARSA algorithm computes the target by integrating the next action:

    .. math:: Q_{target} = r(s, a) + \gamma \sum_{a'} \pi(a')  Q(s', a')

    References
    ----------
    TODO: Find.
    """

    def forward(self, state, action, reward, next_state, done, next_action, policy):
        """Compute the loss and the td-error."""
        pred_q = self.q_function(state, action)
        next_v = self.q_function.value(next_state, policy)
        target_q = reward + self.gamma * next_v * (1 - done)

        return self._build_return(pred_q, target_q)


class SemiGExpectedSARSA(SARSA):
    r"""Implementation of Semi-gradient Expected SARSA algorithm.

    The semi-gradient expected SARSA algorithm computes the target by integrating the
    next action and detaching the gradient.

    .. math:: Q_{target} = (r(s, a) + \gamma \sum_{a'} \pi(a')  Q(s', a')).detach()

    References
    ----------
    TODO: Find.
    """

    def forward(self, state, action, reward, next_state, done, next_action, policy):
        """Compute the loss and the td-error."""
        pred_q = self.q_function(state, action)
        next_v = self.q_function.value(next_state, policy)
        target_q = reward + self.gamma * next_v * (1 - done)

        return self._build_return(pred_q, target_q)


class DExpectedSARSA(SARSA):
    r"""Implementation of Delayed Expected SARSA algorithm.

    The delayed expected SARSA algorithm computes the target by integrating the
    next action over the target q-function.

    .. math:: Q_{target} = (r(s, a) + \gamma \sum_{a'} \pi(a')  Q_{target}(s', a')
    ).detach()

    References
    ----------
    TODO: Find.
    """

    def forward(self, state, action, reward, next_state, done, next_action, policy):
        """Compute the loss and the td-error."""
        pred_q = self.q_function(state, action)
        next_v = self.q_target.value(next_state, policy)
        target_q = reward + self.gamma * next_v * (1 - done)

        return self._build_return(pred_q, target_q)
