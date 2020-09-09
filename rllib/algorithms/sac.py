"""Soft Actor-Critic Algorithm."""

import torch

from rllib.dataset.datatypes import Loss
from rllib.util.neural_networks import DisableGradient
from rllib.util.parameter_decay import Constant, Learnable, ParameterDecay
from rllib.util.utilities import get_entropy_and_log_p, tensor_to_distribution
from rllib.value_function import NNEnsembleQFunction

from .abstract_algorithm import AbstractAlgorithm


class SoftActorCritic(AbstractAlgorithm):
    r"""Implementation of Soft Actor-Critic algorithm.

    SAC is an off-policy policy gradient algorithm.

    epsilon: Learned temperature as a constraint.
    eta: Fixed regularization.
    """

    def __init__(self, eta, regularization=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Actor
        self.target_entropy = -self.policy.dim_action[0]
        assert (
            len(self.policy.dim_action) == 1
        ), "Only Nx1 continuous actions implemented."

        if regularization:  # Regularization: \eta KL(\pi || Uniform)
            if not isinstance(eta, ParameterDecay):
                eta = Constant(eta)
            self.eta = eta
        else:  # Trust-Region: || KL(\pi || Uniform)|| < \epsilon
            if isinstance(eta, ParameterDecay):
                eta = eta()
            self.eta = Learnable(eta, positive=True)

    def post_init(self) -> None:
        """Set derived modules after initialization."""
        super().post_init()
        self.policy.dist_params.update(tanh=True)
        self.policy_target.dist_params.update(tanh=True)

    def actor_loss(self, observation):
        """Get Actor Loss."""
        state = observation.state
        pi = tensor_to_distribution(self.policy(state), **self.policy.dist_params)
        action = pi.rsample()  # re-parametrization trick.

        with DisableGradient(self.critic_target):
            q_val = self.critic_target(state, action)
            if isinstance(self.critic_target, NNEnsembleQFunction):
                q_val = q_val[..., 0]

        entropy, log_p = get_entropy_and_log_p(pi, action, action_scale=1)
        eta_loss = self.eta() * (-log_p - self.target_entropy).detach()
        actor_loss = self.eta().detach() * log_p - q_val

        if self.criterion.reduction == "mean":
            eta_loss, actor_loss = eta_loss.mean(), actor_loss.mean()
        elif self.criterion.reduction == "sum":
            eta_loss, actor_loss = eta_loss.sum(), actor_loss.sum()

        self._info.update(eta=self.eta().detach().item(), entropy=entropy.item())

        return Loss(policy_loss=actor_loss, regularization_loss=eta_loss)

    def get_value_target(self, observation):
        """Get the target of the q function."""
        # Target Q-values
        pi = tensor_to_distribution(
            self.policy(observation.next_state), **self.policy.dist_params
        )
        next_action = self.policy.action_scale * pi.sample()
        next_q = self.critic_target(observation.next_state, next_action)
        if isinstance(self.critic_target, NNEnsembleQFunction):
            next_q = torch.min(next_q, dim=-1)[0]

        _, log_p = get_entropy_and_log_p(pi, next_action, self.policy.action_scale)
        next_v = next_q - self.eta().detach() * log_p
        next_v = next_v * (1.0 - observation.done)
        return self.reward_transformer(observation.reward) + self.gamma * next_v
