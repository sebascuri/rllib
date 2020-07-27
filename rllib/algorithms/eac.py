"""Expected Actor-Critic Algorithm."""
import torch

from rllib.util import discount_sum, integrate, tensor_to_distribution

from .abstract_algorithm import ACLoss
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

    eps = 1e-12

    def forward(self, trajectories):
        """Compute the losses."""
        actor_loss = torch.tensor(0.0)
        critic_loss = torch.tensor(0.0)
        td_error = torch.tensor(0.0)

        for trajectory in trajectories:
            state, action, reward, next_state, done, *r = trajectory

            # ACTOR LOSS
            pi = tensor_to_distribution(self.policy(state))
            if self.policy.discrete_action:
                action = action.long()

            def iq(a):
                return self.critic(state, a) - integrate(
                    lambda a_: self.critic(state, a_), pi, num_samples=self.num_samples
                )

            actor_loss += discount_sum(
                integrate(
                    lambda a: -pi.log_prob(a) * iq(a).detach(),
                    pi,
                    num_samples=self.num_samples,
                ).sum(),
                1,
            )

            # CRITIC LOSS
            with torch.no_grad():
                next_pi = tensor_to_distribution(self.policy(next_state))
                next_v = integrate(lambda a: self.critic_target(next_state, a), next_pi)
                target_q = reward + self.gamma * next_v * (1 - done)

            pred_q = self.critic(state, action)
            critic_loss += self.criterion(pred_q, target_q).mean()
            td_error += (pred_q - target_q).detach().mean()

        num_trajectories = len(trajectories)
        return ACLoss(
            (actor_loss + critic_loss) / num_trajectories,
            actor_loss / num_trajectories,
            critic_loss / num_trajectories,
            td_error / num_trajectories,
        )
