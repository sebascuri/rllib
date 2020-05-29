"""Off-Policy TD Calculation."""
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from rllib.util import integrate, tensor_to_distribution
from rllib.util.neural_networks import deep_copy_module, update_parameters

from .abstract_algorithm import AbstractAlgorithm, TDLoss


class OffPolicyTDLearning(AbstractAlgorithm, metaclass=ABCMeta):
    r"""Implementation of the Off-Policy TD-Learning algorithms.

    The target is computed as:
    .. math:: Q_target(x, a) = Q(x, a) + E_\mu(\sum_{t} \gamma^t \Prod_{s=1}^t c_s td_t,
    where
    ..math:: td_t = r_t + \gamma E_{\pi} Q(x_{t+1}, a) - Q_(x_t, a_t)

    Depending of the choice of the c_s, different algorithms exist. Namely:

    Importance Sampling:
        .. math:: c_s = \pi(a_s|s_s) / \mu(a_s|s_s)

    QLambda:
        .. math:: c_s = \lambda

    TBLambda:
        .. math:: c_s = \lambda \pi(a_s|s_s)

    Retrace
        .. math:: c_s = \lambda \min(1, \pi(a_s|s_s) / \mu(a_s|s_s))


    Parameters
    ----------
    q_function: AbstractQFunction
        Q Function to evaluate.
    policy: AbstractPolicy
        Policy to evaluate.
    criterion: _Loss
        Criterion to optimize.
    gamma: float
        Discount factor.
    lambda_: float.
        Lambda factor for off-policy evaluation.

    References
    ----------
    Precup, D., Sutton, R. S., & Singh, S. (2000).
    Eligibility Traces for Off-Policy Policy Evaluation. ICML.

    Precup, D., Sutton, R. S., & Dasgupta, S. (2001).
    Off-policy temporal-difference learning with function approximation. ICML.

    Geist, M., & Scherrer, B. (2014).
    Off-policy Learning With Eligibility Traces: A Survey. JMLR.

    Harutyunyan, A., Bellemare, M. G., Stepleton, T., & Munos, R. (2016).
    Q (\lambda) with Off-Policy Corrections. ALT.

    Munos, R., Stepleton, T., Harutyunyan, A., & Bellemare, M. (2016).
    Safe and efficient off-policy reinforcement learning. NeuRIPS.

    """

    def __init__(self, q_function, policy, criterion, gamma, lambda_=1):
        super().__init__()
        self.q_function = q_function
        self.q_target = deep_copy_module(q_function)
        self.policy = policy
        self.criterion = criterion
        self.gamma = gamma
        self.lambda_ = lambda_

    @abstractmethod
    def correction(self, pi_log_prob, mu_log_prob):
        """Return the correction at time step t."""
        raise NotImplementedError

    def forward(self, state, action, reward, next_state, done, action_log_prob):
        """Compute the loss and the td-error."""
        n_steps = state.shape[1]
        pred_q = self.q_function(state[:, 0], action[:, 0])
        target_q = pred_q.detach()

        with torch.no_grad():
            discount = 1.
            factor = 1.

            for t in range(n_steps + 1):
                pi = tensor_to_distribution(self.policy(state[:, t]))
                next_pi = tensor_to_distribution(self.policy(next_state[:, t]))
                next_v = integrate(lambda a: self.critic_target(next_state[:, t], a),
                                   next_pi)
                td = reward[:, t] + self.gamma * next_v * done[:, t] - self.q_function(
                    state[:, t], action[:, t])

                target_q += discount * factor * td

                discount *= self.gamma
                factor *= self.correction(pi.log_prob(action[:, t]),
                                          action_log_prob[:, t])

        return self._build_return(pred_q, target_q)

    def _build_return(self, pred_q, target_q):
        return TDLoss(loss=self.criterion(pred_q, target_q),
                      td_error=(pred_q - target_q).detach())

    def update(self):
        """Update the target network."""
        update_parameters(self.q_target, self.q_function,
                          tau=self.value_function.tau)


class ImportanceSamplingOffPolicyTD(nn.Module, metaclass=ABCMeta):
    r"""Importance Sampling Off-Policy TD-Learning algorithm.

    The correction factor is given by:

    .. math:: c_s = \pi(a_s|s_s) / \mu(a_s|s_s)

    References
    ----------
    Precup, D., Sutton, R. S., & Dasgupta, S. (2001).
    Off-policy temporal-difference learning with function approximation. ICML.

    Geist, M., & Scherrer, B. (2014).
    Off-policy Learning With Eligibility Traces: A Survey. JMLR.

    """

    def correction(self, pi_log_prob, mu_log_prob):
        """Return the correction at time step t."""
        return torch.exp(pi_log_prob - mu_log_prob)


class TDLambda(nn.Module, metaclass=ABCMeta):
    r"""TD-Lambda algorithm.

    The correction factor is given by:
    .. math:: c_s = \lambda

    References
    ----------
    Harutyunyan, A., Bellemare, M. G., Stepleton, T., & Munos, R. (2016).
    Q (\lambda) with Off-Policy Corrections. ALT.

    """

    def correction(self, pi_log_prob, mu_log_prob):
        """Return the correction at time step t."""
        return self._lambda


class TreeBackupLambda(nn.Module, metaclass=ABCMeta):
    r"""Tree-Backup Lambda Off-Policy TD-Learning algorithm.

    The correction factor is given by:
    .. math:: c_s = \lambda * \pi(a_s | s_s)

    References
    ----------
    Precup, D., Sutton, R. S., & Singh, S. (2000).
    Eligibility Traces for Off-Policy Policy Evaluation. ICML.

    """

    def correction(self, pi_log_prob, mu_log_prob):
        """Return the correction at time step t."""
        return self._lambda * pi_log_prob


class Retrace(nn.Module, metaclass=ABCMeta):
    r"""Importance Sampling Off-Policy TD-Learning algorithm.

    .. math:: c_s = \lambda min(1, \pi(a_s|s_s) / \mu(a_s|s_s))

    References
    ----------
    Harutyunyan, A., Bellemare, M. G., Stepleton, T., & Munos, R. (2016).
    Q (\lambda) with Off-Policy Corrections. ALT.

    """

    def correction(self, pi_log_prob, mu_log_prob):
        """Return the correction at time step t."""
        return self._lambda * torch.exp(pi_log_prob - mu_log_prob).clamp_max(1.)
