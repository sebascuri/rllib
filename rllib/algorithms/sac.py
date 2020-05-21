"""Soft Actor-Critic Algorithm."""

import torch

from rllib.util.utilities import tensor_to_distribution
from rllib.util.neural_networks import deep_copy_module, disable_gradient, \
    update_parameters
from .abstract_algorithm import AbstractAlgorithm, ACLoss, TDLoss


class SoftActorCritic(AbstractAlgorithm):
    r"""Implementation of Soft Actor-Critic algorithm.

    SAC is an off-policy policy gradient algorithm.
    """

    eps = 1e-12

    def __init__(self, policy, critic, criterion, temperature, gamma):
        super().__init__()
        # Actor
        self.policy = policy
        self.policy_target = deep_copy_module(policy)

        # Critic
        self.q_function = critic
        self.q_target = deep_copy_module(critic)

        self.temperature = temperature

        self.criterion = criterion
        self.gamma = gamma

    def actor_loss(self, state):
        """Get Actor Loss."""
        pi = tensor_to_distribution(self.policy(state), tanh=True,
                                    action_scale=self.policy.action_scale)
        if pi.has_rsample:
            action = pi.rsample()  # re-parametrization trick.
        else:
            action = pi.sample()

        with disable_gradient(self.q_target):
            q_val = self.q_target(state, action)
            if type(q_val) is list:
                q_val = torch.min(*q_val)

        return (self.temperature * pi.log_prob(action) - q_val).mean()

    def critic_loss(self, state, action, reward, next_state, done):
        """Get Critic Loss and td-error."""
        pred_q = self.q_function(state, action)
        if type(pred_q) is not list:
            pred_q = [pred_q]

        # Target Q-values
        with torch.no_grad():
            pi = tensor_to_distribution(self.policy(next_state), tanh=True,
                                        action_scale=self.policy.action_scale)
            next_action = pi.sample()
            next_q = self.q_target(next_state, next_action)
            if type(next_q) is list:
                next_q = torch.min(*next_q)

            next_v = next_q - self.temperature * pi.log_prob(next_action)
            target_q = reward + self.gamma * next_v * (1 - done)

        critic_loss = torch.zeros_like(target_q)
        td_error = torch.zeros_like(target_q)
        for q in pred_q:
            critic_loss += self.criterion(q, target_q)
            td_error += (q - target_q).detach()

        return TDLoss(critic_loss, td_error)

    def forward(self, state, action, reward, next_state, done):
        """Compute the losses."""
        # Critic Loss
        critic_loss, td_error = self.critic_loss(state, action, reward, next_state,
                                                 done)

        # Actor loss
        actor_loss = self.actor_loss(state)
        return ACLoss(actor_loss, critic_loss, td_error)

    def update(self):
        """Update the baseline network."""
        update_parameters(self.q_target, self.q_function, tau=self.q_function.tau)
        update_parameters(self.policy_target, self.policy, tau=self.policy.tau)
