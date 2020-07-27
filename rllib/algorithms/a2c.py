"""Advantage Actor Critic Algorithm."""
from rllib.util.utilities import integrate, tensor_to_distribution

from .ac import ActorCritic


class A2C(ActorCritic):
    r"""Implementation of Advantage Actor-Critic (A2C) algorithm.

    A2C is an on-policy model-free control algorithm.
    A2C computes the policy gradient using an advantage function to estimate the returns
    (sum of discounted rewards).

    The A2C algorithm is a policy gradient algorithm that estimates the
    gradient:
    .. math:: \grad J = \int_{\tau} \grad \log \pi(s_t) A(s_t, a_t),
    where the previous integral is computed through samples (s_t, a_t) samples and the
    advantage function is calculated as:
    .. math:: A(s, a) = Q(s, a) - V(s), V(s) = \int_{a} Q(s, a)


    Parameters
    ----------
    policy : AbstractPolicy
        Policy to optimize.
    critic : AbstractQFunction
        Critic that evaluates the current policy.
    criterion : _Loss
        Criterion to optimize the baseline.
    gamma : float
        Discount factor.

    References
    ----------
    Mnih, V., et al. (2016).
    Asynchronous methods for deep reinforcement learning. ICML.
    """

    def __init__(self, policy, critic, criterion, gamma):
        super().__init__(policy, critic, criterion, gamma)

    def returns(self, trajectory):
        """Estimate the returns of a trajectory."""
        state, action = trajectory.state, trajectory.action
        pred_q = self.critic(state, action)
        pi = tensor_to_distribution(self.policy(state))
        returns = pred_q - integrate(
            lambda a: self.critic(state, a), pi, num_samples=self.num_samples
        )
        return returns
