"""Expected SARSA Algorithm."""

import torch
import torch.nn as nn
from rllib.util.utilities import integrate
import copy
from .q_learning import QLearningLoss


class ExpectedSARSA(nn.Module):
    r"""Implementation of Expected SARSA algorithm.

    SARSA is an on-policy model-free control algorithm
    The expected SARSA algorithm computes the target by integrating the next action:

    .. math:: Q_{target} = r(s, a) + \gamma \sum_{a'} \pi(a')  Q(s', a')

    References
    ----------
    TODO: Find.
    """
    def __init__(self, q_function, criterion, policy, gamma):
        super().__init__()
        self.q_function = q_function
        self.q_target = copy.deepcopy(q_function)
        self.policy = policy
        self.criterion = criterion
        self.gamma = gamma

    def _build_return(self, pred_q, target_q):
        return QLearningLoss(loss=self.criterion(pred_q, target_q),
                             td_error=(pred_q - target_q).detach())

    def forward(self, state, action, reward, next_state, done):
        """Compute the loss and the td-error."""
        pred_q = self.q_function(state, action)
        with torch.no_grad():
            next_v = integrate(lambda a: self.q_target(state, a), self.policy(state))
            target_q = reward + self.gamma * next_v * (1 - done)

        return self._build_return(pred_q, target_q)

    def update(self):
        self.q_target.update_parameters(self.q_function.parameters())


class GradientExpectedSARSA(ExpectedSARSA):
    r"""Implementation of Gradient-Expected SARSA algorithm.

    The semi-gradient expected SARSA algorithm computes the target by integrating the
    next action and detaching the gradient.

    .. math:: Q_{target} = (r(s, a) + \gamma \sum_{a'} \pi(a')  Q(s', a')).detach()

    References
    ----------
    TODO: Find.
    """

    def forward(self, state, action, reward, next_state, done):
        """Compute the loss and the td-error."""
        pred_q = self.q_function(state, action)
        next_v = integrate(lambda a: self.q_function(state, a), self.policy(state))
        target_q = reward + self.gamma * next_v * (1 - done)

        return self._build_return(pred_q, target_q)
