"""Abstract calculation of TD-Target."""
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from rllib.util.neural_networks.utilities import reverse_cumsum
from rllib.util.utilities import get_entropy_and_log_p, tensor_to_distribution
from rllib.value_function import AbstractValueFunction, IntegrateQValueFunction


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
        self.value_target = IntegrateQValueFunction(critic, policy, num_samples)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_samples = num_samples

    @abstractmethod
    def correction(self, pi_log_p, behavior_log_p):
        """Return the correction at time step t."""
        raise NotImplementedError

    def forward(self, observation):
        """Compute the loss and the td-error."""
        state, action, reward, next_state, done, *_ = observation
        behavior_log_p = observation.log_prob_action
        n_steps = state.shape[1]

        # done_t indicates if the current state is done.
        done_t = torch.cat((torch.zeros(done.shape[0], 1), done), -1)[:, :-1]

        # Compute off-policy correction factor.
        if self.policy is not None:
            pi = tensor_to_distribution(self.policy(state), **self.policy.dist_params)
            _, log_p = get_entropy_and_log_p(pi, action, self.policy.action_scale)
        else:
            log_p = behavior_log_p
        correction = self.correction(log_p, behavior_log_p)

        # Compute Q(state, action) and \E_\pi[Q(next_state, \pi(next_state)].
        if isinstance(self.critic, AbstractValueFunction):
            this_v = self.critic(state) * (1.0 - done_t)
            next_v = self.critic(next_state)
        else:
            this_v = self.critic(state, action) * (1.0 - done_t)

            if self.policy is not None:
                next_v = self.value_target(next_state)
            else:
                next_v = self.critic(next_state[:, : n_steps - 1], action[:, 1:])
                last_v = torch.zeros(next_v.shape[0], 1)
                if last_v.ndim < next_v.ndim:
                    last_v = last_v.unsqueeze(-1).repeat_interleave(
                        next_v.shape[-1], -1
                    )
                next_v = torch.cat((next_v, last_v), -1)
        next_v = next_v * (1.0 - done)
        # Compute td = r + gamma E\pi[Q(next_state, \pi(next_state)] - Q(state, action).
        td = self.td(this_v, next_v, reward, correction)

        # Compute correction factor_t = \Prod_{i=1,t} c_i.
        correction_factor = torch.cumprod(correction, dim=-1)

        # Compute discount_t = \gamma ** (t-1)
        discount = torch.pow(torch.tensor(self.gamma), torch.arange(n_steps))

        # Compute target = Q(s, a) + \sum_{i=1,t} discount_i factor_i td_i. See RETRACE.
        target = this_v + reverse_cumsum(td * discount * correction_factor)

        return target

    def td(self, this_v, next_v, reward, correction):
        """Compute the TD error."""
        return reward + self.gamma * next_v - this_v
