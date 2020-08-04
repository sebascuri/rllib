"""Abstract calculation of TD-Target."""
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from rllib.util import integrate, tensor_to_distribution
from rllib.value_function.abstract_value_function import AbstractValueFunction


class AbstractTDTarget(nn.Module, metaclass=ABCMeta):
    r"""Abstract implementation of the target for Off-Policy TD-Learning algorithms.

    The target is computed as:
    .. math:: Q_target(x, a) = Q(x, a) + E_\mu(\sum_{t} \gamma^t \Prod_{s=1}^t c_s td_t,
    where
    ..math:: td_t = \rho_t(r_t + \gamma E_{\pi} Q(x_{t+1}, a) - Q_(x_t, a_t))

    Depending of the choice of the c_s, different algorithms exist. Namely:

    Importance Sampling:
        .. math:: c_s = \pi(a_s|s_s) / \mu(a_s|s_s)
        .. math:: \rho_s = 1

    QLambda:
        .. math:: c_s = \lambda
        .. math:: \rho_s = 1

    TBLambda:
        .. math:: c_s = \lambda \pi(a_s|s_s)
        .. math:: \rho_s = 1

    ReTrace
        .. math:: c_s = \lambda \min(1, \pi(a_s|s_s) / \mu(a_s|s_s))
        .. math:: \rho_s = 1

    VTrace
        .. math:: c_s = \lambda \min(1, \pi(a_s|s_s) / \mu(a_s|s_s))
        .. math:: \rho_s = \min(\rho_bar, \pi(a_s|s_s) / \mu(a_s|s_s))

    Parameters
    ----------
    critic: AbstractQFunction
        Q/Value Function to evaluate.
    policy: AbstractPolicy, optional
        Policy to evaluate. If None, then on-policy data is assumed.
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

    def __init__(self, critic, policy=None, gamma=0.99, lambda_=1.0, num_samples=15):
        super().__init__()
        self.critic = critic
        self.policy = policy
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_samples = num_samples

    @abstractmethod
    def correction(self, pi_log_prob, mu_log_prob):
        """Return the correction at time step t."""
        raise NotImplementedError

    def forward(self, observation):
        """Compute the loss and the td-error."""
        state, action, reward, next_state, done, *_ = observation
        log_prob_action = observation.log_prob_action

        n_steps = state.shape[1]

        if isinstance(self.critic, AbstractValueFunction):
            target = self.critic(state[:, 0])
        else:
            target = self.critic(state[:, 0], action[:, 0])

        discount = 1.0
        factor = 1.0
        done_t = torch.zeros_like(done[:, 0])  # Early truncation.

        for t in range(n_steps):
            if self.policy is not None:
                pi = tensor_to_distribution(self.policy(state[:, t]))
                eval_log_prob = pi.log_prob(action[:, t])
            else:
                eval_log_prob = log_prob_action[:, t]

            correction = self.correction(eval_log_prob, log_prob_action[:, t])

            if isinstance(self.critic, AbstractValueFunction):
                this_v = self.critic(state[:, t]) * (1.0 - done_t)
                next_v = self.critic(next_state[:, t]) * (1.0 - done[:, t])
            else:
                this_v = self.critic(state[:, t], action[:, t]) * (1.0 - done_t)

                if self.policy is not None:
                    next_pi = tensor_to_distribution(self.policy(next_state[:, t]))
                    next_v = integrate(
                        lambda a: self.critic(next_state[:, t], a),
                        next_pi,
                        num_samples=self.num_samples,
                    ) * (1.0 - done[:, t])
                else:  # Use sampled next action. This allows to compute n-step returns.
                    if t < n_steps - 1:
                        next_v = self.critic(next_state[:, t], action[:, t + 1]) * (
                            1.0 - done[:, t]
                        )
                    else:
                        next_v = torch.zeros_like(target)

            target += (
                discount * factor * self.td(this_v, next_v, reward[:, t], correction)
            )

            discount *= self.gamma
            factor *= correction
            done_t = 1.0 - (1.0 - done_t) * (1.0 - done[:, t])

        return target

    def td(self, this_v, next_v, reward, correction):
        """Compute the TD error."""
        return reward + self.gamma * next_v - this_v
