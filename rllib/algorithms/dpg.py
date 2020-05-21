"""Deterministic Policy Gradient Algorithm."""

import torch

from rllib.util.neural_networks.utilities import disable_gradient, deep_copy_module, \
    update_parameters
from rllib.util.utilities import tensor_to_distribution
from .abstract_algorithm import AbstractAlgorithm, ACLoss, TDLoss


class DPG(AbstractAlgorithm):
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
        self.q_target = deep_copy_module(q_function)
        self.criterion = criterion

        # Actor
        self.policy = policy
        self.policy_target = deep_copy_module(policy)

        self.gamma = gamma
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

    def _add_noise(self, action):
        next_noise = (torch.randn_like(action) * self.policy_noise).clamp(
            -self.noise_clip, self.noise_clip)
        return (action + next_noise).clamp(-1, 1)

    def actor_loss(self, state):
        """Get Actor Loss."""
        action = tensor_to_distribution(self.policy(state)).mean.clamp(-1, 1)
        with disable_gradient(self.q_function):
            q = self.q_function(state, action)
            if type(q) is list:
                q = q[0]
        return -q

    def critic_loss(self, state, action, reward, next_state, done):
        """Get Critic Loss and td-error."""
        pred_q = self.q_function(state, action)
        if type(pred_q) is not list:
            pred_q = [pred_q]

        # Target Q-values
        with torch.no_grad():
            next_action = self._add_noise(
                tensor_to_distribution(self.policy_target(next_state)).sample())
            next_v = self.q_target(next_state, next_action)
            if type(next_v) is list:
                next_v = torch.min(*next_v)
            target_q = reward + self.gamma * next_v * (1 - done)

        critic_loss = torch.zeros_like(target_q)
        td_error = torch.zeros_like(target_q)
        for q in pred_q:
            critic_loss += (self.criterion(q, target_q))
            td_error += q.detach() - target_q.detach()

        return TDLoss(critic_loss, td_error)

    def forward(self, state, action, reward, next_state, done):
        """Compute the losses and the td-error."""
        # Critic Loss
        critic_loss, td_error = self.critic_loss(state, action, reward, next_state,
                                                 done)

        # Actor loss
        actor_loss = self.actor_loss(state)

        return ACLoss(actor_loss=actor_loss.squeeze(-1),
                      critic_loss=critic_loss.squeeze(-1),
                      td_error=td_error.squeeze(-1))

    def update(self):
        """Update the target network."""
        update_parameters(self.policy_target, self.policy, tau=self.policy.tau)
        update_parameters(self.q_target, self.q_function, tau=self.q_function.tau)
