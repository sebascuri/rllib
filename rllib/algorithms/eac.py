"""Expected Actor-Critic Algorithm."""
import torch
import torch.nn as nn
import copy
from rllib.util.utilities import integrate
from .ac import PGLoss


class ActorCritic(nn.Module):
    r"""Implementation of Policy Gradient algorithm.

    Policy-Gradient is an on-policy model-free control algorithm.
    Policy-Gradient computes the policy gradient using a critic to estimate the returns
    (sum of discounted rewards).

    The Policy-Gradient algorithm is a policy gradient algorithm that estimates the
    gradient:
    .. math:: \grad J = \int_{\tau} \grad \log \pi(s_t) Q(s_t, a_t),
    where the previous integral is computed through samples (s_t, a_t) samples.


    Parameters
    ----------
    policy: AbstractPolicy
        Policy to optimize.
    critic: AbstractValueFunction
        Critic that evaluates the current policy.
    criterion: _Loss
        Criterion to optimize the baseline.
    gamma: float
        Discount factor.

    References
    ----------
    Sutton, R. S., McAllester, D. A., Singh, S. P., & Mansour, Y. (2000).
    Policy gradient methods for reinforcement learning with function approximation.NIPS.

    Konda, V. R., & Tsitsiklis, J. N. (2000).
    Actor-critic algorithms. NIPS.
    """

    def __init__(self, policy, critic, criterion, gamma):
        super().__init__()
        # Actor
        self.policy = policy
        self.policy_target = copy.deepcopy(policy)

        # Critic
        self.critic = critic
        self.critic_target = copy.deepcopy(critic)

        self.criterion = criterion
        self.gamma = gamma

    def forward(self, trajectories):
        """Compute the losses."""
        actor_loss = torch.tensor(0.)
        critic_loss = torch.tensor(0.)

        for trajectory in trajectories:
            state, action, reward, next_state, done, *r = trajectory

            # ACTOR LOSS
            pi = self.policy(state)
            if self.policy.discrete_action:
                action = action.long()

            actor_loss += integrate(lambda a: -pi.log_prob(a) * self.critic(state, a),
                                    pi).sum()

            # CRITIC LOSS
            with torch.no_grad():
                next_v = integrate(lambda a: self.critic_target(next_state, a),
                                   self.policy(next_state))
                target_q = reward + self.gamma * next_v * (1 - done)

            critic_loss += self.criterion(self.critic(state, action), target_q)

        return PGLoss(actor_loss, critic_loss)

    def update(self):
        """Update the baseline network."""
        self.policy_target.update_parameters(self.policy.parameters())
        self.critic_target.update_parameters(self.critic.parameters())
