"""Deterministic Policy Gradient Algorithm."""
import torch
import torch.nn as nn
import copy
from collections import namedtuple

DPGLoss = namedtuple('DPGLoss', ['actor_loss', 'critic_loss', 'td_error'])


class DPG(nn.Module):
    r"""Implementation of DPG algorithm.

    DPG is an off-policy model-free control algorithm.

    The DPG algorithm is an actor-critic algorithm that has a policy that estimates:
    .. math:: a = \pi(s) = \argmax_a Q(s, a)


    Parameters
    ----------
    q_function: AbstractQFunction
        Q_function to optimize.
    criterion: _Loss
        Criterion to optimize.
    gamma: float
        Discount factor.

    References
    ----------
    Silver, David, et al. (2014)
    Deterministic policy gradient algorithms. JMLR.
    Lillicrap et. al. (2016).
    Continuous Control with Deep Reinforcement Learning. ICLR.
    """

    def __init__(self, q_function, policy, criterion, gamma, policy_noise, noise_clip):
        super().__init__()
        # Critic
        self.q_function = q_function
        self.q_target = copy.deepcopy(q_function)
        self.criterion = criterion

        # Actor
        self.policy = policy
        self.policy_target = copy.deepcopy(policy)

        self.gamma = gamma
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

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
        pred_q = self.q_function(state, action)
        if type(pred_q) is not list:
            pred_q = [pred_q]

        # target = r + gamma * Q(x', \pi(x'))
        next_action = self._add_noise(self.policy_target(next_state).rsample())
        next_v = self.q_target(next_state, next_action)
        if type(next_v) is list:
            next_v = torch.min(*next_v)
        target_q = reward + self.gamma * next_v * (1 - done)

        critic_loss = torch.zeros_like(target_q)
        td_error = torch.zeros_like(target_q)
        for q in pred_q:
            critic_loss += (self.criterion(q, target_q))
            td_error += q.detach() - target_q.detach()

        # Actor loss
        action = self.policy(state).mean.clamp(-1, 1)
        q = self.q_function(state.float(), action)
        if type(q) is list:
            q = q[0]
        actor_loss = -q

        return DPGLoss(actor_loss=actor_loss, critic_loss=critic_loss,
                       td_error=td_error)

    def update(self):
        """Update the target network."""
        self.q_target.update_parameters(self.q_function.parameters())
        self.policy_target.update_parameters(self.policy.parameters())
