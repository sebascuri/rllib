"""Actor-Critic Algorithm."""

from collections import namedtuple

import torch
import torch.nn as nn

from rllib.util import integrate, discount_sum, tensor_to_distribution
from rllib.util.neural_networks import deep_copy_module, update_parameters


PGLoss = namedtuple('PolicyGradientLoss',
                    ['actor_loss', 'critic_loss', 'td_error'])


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
    critic: AbstractQFunction
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

    eps = 1e-12

    def __init__(self, policy, critic, criterion, gamma):
        super().__init__()
        # Actor
        self.policy = policy
        self.policy_target = deep_copy_module(policy)

        # Critic
        self.critic = critic
        self.critic_target = deep_copy_module(critic)

        self.criterion = criterion
        self.gamma = gamma

    def returns(self, trajectory):
        """Estimate the returns of a trajectory."""
        state, action = trajectory.state, trajectory.action
        pred_q = self.critic(state, action)
        returns = pred_q
        return returns

    def forward(self, trajectories):
        """Compute the losses."""
        actor_loss = torch.tensor(0.)
        critic_loss = torch.tensor(0.)
        td_error = torch.tensor(0.)

        for trajectory in trajectories:
            state, action, reward, next_state, done, *r = trajectory

            # ACTOR LOSS
            pi = tensor_to_distribution(self.policy(state))
            if self.policy.discrete_action:
                action = action.long()
            with torch.no_grad():
                returns = self.returns(trajectory)
                returns = (returns - returns.mean()) / (returns.std() + self.eps)

            actor_loss += discount_sum(-pi.log_prob(action) * returns, self.gamma)

            # CRITIC LOSS
            with torch.no_grad():
                next_pi = tensor_to_distribution(self.policy(next_state))
                next_v = integrate(lambda a: self.critic_target(next_state, a), next_pi)
                target_q = reward + self.gamma * next_v * (1 - done)

            pred_q = self.critic(state, action)
            critic_loss += self.criterion(pred_q, target_q).mean()
            td_error += (pred_q - target_q).detach().mean()

        num_trajectories = len(trajectories)
        return PGLoss(actor_loss / num_trajectories, critic_loss / num_trajectories,
                      td_error / num_trajectories)

    def update(self):
        """Update the baseline network."""
        update_parameters(self.policy_target, self.policy, tau=self.policy.tau)
        update_parameters(self.critic_target, self.critic, tau=self.critic.tau)
