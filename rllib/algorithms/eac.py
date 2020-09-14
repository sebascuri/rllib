"""Expected Actor-Critic Algorithm."""

from rllib.dataset.datatypes import Loss
from rllib.util.utilities import (
    get_entropy_and_log_p,
    integrate,
    tensor_to_distribution,
)

from .ac import ActorCritic


class ExpectedActorCritic(ActorCritic):
    r"""Implementation of Expected Policy Gradient algorithm.

    EPG is an on-policy model-free control algorithm.
    EPG computes the policy gradient using a critic to estimate the returns
    (sum of discounted rewards).

    The EPG algorithm is a policy gradient algorithm that estimates the
    gradient:
    .. math:: \grad J = \int_{\tau} \grad \log \pi(s_t) Q(s_t, a_t) - V(s_t),
    where the previous integral is computed through samples (s_t) and exactly
    integrating the actions with the policy.


    References
    ----------
    Ciosek, K., & Whiteson, S. (2018).
    Expected policy gradients. AAAI.
    """

    def actor_loss(self, observation):
        """Get Actor loss."""
        state, action, *_ = observation

        pi = tensor_to_distribution(self.policy(state), **self.policy.dist_params)
        entropy, _ = get_entropy_and_log_p(pi, action, self.policy.action_scale)

        policy_loss = integrate(
            lambda a: -pi.log_prob(a)
            * (
                self.critic(state, self.policy.action_scale * a)
                - self.value_target(state)
            ).detach(),
            pi,
            num_samples=self.num_samples,
        ).sum()

        return Loss(policy_loss=policy_loss).reduce(self.criterion.reduction)
