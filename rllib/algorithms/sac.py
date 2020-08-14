"""Soft Actor-Critic Algorithm."""

import torch

from rllib.util.neural_networks import (
    DisableGradient,
    deep_copy_module,
    update_parameters,
)
from rllib.util.parameter_decay import Constant, Learnable, ParameterDecay
from rllib.util.utilities import tensor_to_distribution
from rllib.value_function import NNEnsembleQFunction

from .abstract_algorithm import AbstractAlgorithm, SACLoss, TDLoss


class SoftActorCritic(AbstractAlgorithm):
    r"""Implementation of Soft Actor-Critic algorithm.

    SAC is an off-policy policy gradient algorithm.

    epsilon: Learned temperature as a constraint.
    eta: Fixed regularization.
    """

    def __init__(
        self, q_function, criterion, eta, regularization=False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        # Actor
        self.target_entropy = -self.policy.dim_action[0]
        assert (
            len(self.policy.dim_action) == 1
        ), "Only Nx1 continuous actions implemented."

        # Critic
        self.q_function = q_function
        self.q_target = deep_copy_module(q_function)

        if regularization:  # Regularization: \eta KL(\pi || Uniform)
            if not isinstance(eta, ParameterDecay):
                eta = Constant(eta)
            self.eta = eta
        else:  # Trust-Region: || KL(\pi || Uniform)|| < \epsilon
            self.eta = Learnable(eta, positive=True)

        self.criterion = criterion

    def actor_loss(self, state):
        """Get Actor Loss."""
        pi = tensor_to_distribution(self.policy(state), tanh=True)
        action = pi.rsample()  # re-parametrization trick.

        with DisableGradient(self.q_target):
            q_val = self.q_target(state, action)
            if isinstance(self.q_function, NNEnsembleQFunction):
                q_val = q_val[..., 0]

        log_prob = pi.log_prob(action)
        eta_loss = self.eta() * (-log_prob - self.target_entropy).detach()
        actor_loss = self.eta().detach() * log_prob - q_val

        if self.criterion.reduction == "mean":
            eta_loss, actor_loss = eta_loss.mean(), actor_loss.mean()
        elif self.criterion.reduction == "sum":
            eta_loss, actor_loss = eta_loss.sum(), actor_loss.sum()

        return actor_loss, eta_loss

    def get_q_target(self, observation):
        """Get the target of the q function."""
        # Target Q-values
        pi = tensor_to_distribution(self.policy(observation.next_state), tanh=True)
        next_action = pi.sample()
        next_q = self.q_target(observation.next_state, next_action)
        if isinstance(self.q_target, NNEnsembleQFunction):
            next_q = torch.min(next_q, dim=-1)[0]

        log_prob = pi.log_prob(next_action)
        next_v = next_q - self.eta().detach() * log_prob
        next_v = next_v * (1.0 - observation.done)
        return self.reward_transformer(observation.reward) + self.gamma * next_v

    def critic_loss(self, state, action, q_target):
        """Get Critic Loss and td-error."""
        pred_q = self.q_function(state, action)
        if isinstance(self.q_function, NNEnsembleQFunction):
            q_target = q_target.unsqueeze(-1).repeat_interleave(
                self.q_function.num_heads, -1
            )

        critic_loss = self.criterion(pred_q, q_target)
        td_error = pred_q.detach() - q_target.detach()

        if isinstance(self.q_function, NNEnsembleQFunction):
            critic_loss = critic_loss.sum(-1)
            td_error = td_error.sum(-1)

        return TDLoss(critic_loss, td_error)

    def forward(self, observation):
        """Compute the losses."""
        state, action, reward, next_state, done, *r = observation
        # Critic Loss
        with torch.no_grad():
            q_target = self.get_q_target(observation)
        critic_loss, td_error = self.critic_loss(
            state, action / self.policy.action_scale, q_target
        )

        # Actor loss
        policy_loss, eta_loss = self.actor_loss(state)
        self._info = {"eta": self.eta().detach().item()}

        return SACLoss(
            loss=policy_loss + critic_loss + critic_loss + eta_loss,
            policy_loss=policy_loss,
            critic_loss=critic_loss,
            eta_loss=eta_loss,
            td_error=td_error,
        )

    def update(self):
        """Update the baseline network."""
        update_parameters(self.q_target, self.q_function, tau=self.q_function.tau)
