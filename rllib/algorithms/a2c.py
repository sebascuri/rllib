"""Advantage Actor Critic Algorithm."""

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

    References
    ----------
    Mnih, V., et al. (2016).
    Asynchronous methods for deep reinforcement learning. ICML.
    """

    def returns(self, trajectory):
        """Estimate the returns of a trajectory."""
        state, action = trajectory.state, trajectory.action
        weight = self.get_ope_weight(state, action, trajectory.log_prob_action)
        return weight * (self.critic(state, action) - self.value_target(state))
