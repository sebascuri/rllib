"""Deterministic Policy Gradient Algorithm."""
import torch
import torch.nn as nn
import copy
from collections import namedtuple
from .sarsa import SARSA

DPGLoss = namedtuple('DPGLoss', ['actor_loss', 'critic_loss', 'td_error'])


class DPG(nn.Module):
    r"""Implementation of DPG algorithm.

    DPG is an off-policy model-free control algorithm.

    The DPG algorithm is an actor-critic algorithm that has a policy that estimates:
    .. math:: a = \pi(s) = \argmax_a Q(s, a)


    Parameters
    ----------
    q_function: AbstractQFunction
        q_function to optimize.
    criterion: _Loss
        Criterion to optimize.
    gamma: float
        discount factor.

    References
    ----------
    Silver, David, et al. (2014) "Deterministic policy gradient algorithms." JMLR.
    """

    def __init__(self, sarsa_algorithm: SARSA, q_function, policy, criterion, gamma):
        super().__init__()
        self.sarsa_algorithm = sarsa_algorithm(q_function, criterion, gamma)
        self.policy = policy
        self.policy_target = copy.deepcopy(policy)
        self.criterion = criterion
        self.gamma = gamma

    def _add_noise(self, action):
        next_noise = (torch.randn_like(action) * self.policy_noise).clamp(
            -self.noise_clip, self.noise_clip)
        return (action + next_noise).clamp(-1, 1)

    def _actor_loss(self, state):
        action = self.policy(state).mean.clamp(-1, 1)
        return -self.q_function(state.float(), action)

    def forward(self, state, action, reward, next_state, done):
        """Compute the losses and the td-error."""
        # Critic Loss
        next_action = self._add_noise(self.policy_target(next_state).rsample())
        critic = self.sarsa_algorithm(
            state, action, reward, next_state, done, next_action)

        # Actor loss
        action = self.policy(state).mean.clamp(-1, 1)
        actor_loss = -self.sarsa_algorithm.q_function(state.float(), action)

        return DPGLoss(actor_loss=actor_loss, critic_loss=critic.loss,
                       td_error=critic.td_error)

    def update(self):
        """Update the target network."""
        self.sarsa_algorithm.update()
        self.policy_target.update_parameters(self.policy.parameters())
