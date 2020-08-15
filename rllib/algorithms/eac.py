"""Expected Actor-Critic Algorithm."""

from rllib.util import (
    discount_sum,
    get_entropy_and_logp,
    integrate,
    tensor_to_distribution,
)

from .abstract_algorithm import Loss
from .ac import ActorCritic


class ExpectedActorCritic(ActorCritic):
    r"""Implementation of Expected Policy Gradient algorithm.

    EPG is an on-policy model-free control algorithm.
    EPG computes the policy gradient using a critic to estimate the returns
    (sum of discounted rewards).

    The EPG algorithm is a policy gradient algorithm that estimates the
    gradient:
    .. math:: \grad J = \int_{\tau} \grad \log \pi(s_t) Q(s_t, a_t),
    where the previous integral is computed through samples (s_t) and exactly
    integrating the actions with the policy.


    References
    ----------
    Ciosek, K., & Whiteson, S. (2018).
    Expected policy gradients. AAAI.
    """

    def actor_loss(self, observation):
        """Get Actor loss."""
        state, action, reward, next_state, done, *r = observation

        pi = tensor_to_distribution(self.policy(state))
        entropy, log_p = get_entropy_and_logp(pi, action)

        def int_q(a, s=state, pi_=pi):
            """Integrate the critic w.r.t. the action."""
            return self.critic(s, a) - integrate(
                lambda a_: self.critic(s, a_), pi_, num_samples=self.num_samples
            )

        policy_loss = discount_sum(
            integrate(
                lambda a, pi_=pi, iq=int_q: -pi_.log_prob(a) * iq(a).detach(),
                pi,
                num_samples=self.num_samples,
            ).sum(),
            1,
        )

        return Loss(
            loss=policy_loss, policy_loss=policy_loss, regularization_loss=-entropy
        )
