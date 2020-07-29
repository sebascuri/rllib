"""Soft Actor-Critic Algorithm."""

import torch

from rllib.util.neural_networks import (
    deep_copy_module,
    disable_gradient,
    update_parameters,
)
from rllib.util.parameter_decay import Constant, Learnable, ParameterDecay
from rllib.util.utilities import RewardTransformer, tensor_to_distribution
from rllib.util.value_estimation import mb_return
from rllib.value_function import IntegrateQValueFunction, NNEnsembleQFunction

from .abstract_algorithm import AbstractAlgorithm, SACLoss, TDLoss


class SoftActorCritic(AbstractAlgorithm):
    r"""Implementation of Soft Actor-Critic algorithm.

    SAC is an off-policy policy gradient algorithm.

    epsilon: Learned temperature as a constraint.
    eta: Fixed regularization.
    """

    def __init__(
        self,
        policy,
        q_function,
        criterion,
        gamma,
        eta,
        regularization=False,
        reward_transformer=RewardTransformer(),
    ):
        super().__init__()
        # Actor
        self.policy = policy
        self.target_entropy = -policy.dim_action

        # Critic
        self.q_function = q_function
        self.q_target = deep_copy_module(q_function)

        if regularization:  # Regularization: \eta KL(\pi || Uniform)
            if not isinstance(eta, ParameterDecay):
                eta = Constant(eta)
            self.eta = eta
        else:  # Trust-Region: || KL(\pi || Uniform)|| < \epsilon
            self.eta = Learnable(eta, positive=True)

        self.reward_transformer = reward_transformer

        self.criterion = criterion
        self.gamma = gamma

    def actor_loss(self, state):
        """Get Actor Loss."""
        pi = tensor_to_distribution(self.policy(state), tanh=True)
        action = pi.rsample()  # re-parametrization trick.

        with disable_gradient(self.q_target):
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

    def get_q_target(self, reward, next_state, done):
        """Get the target of the q function."""
        # Target Q-values
        pi = tensor_to_distribution(self.policy(next_state), tanh=True)
        next_action = pi.sample()
        next_q = self.q_target(next_state, next_action)
        if isinstance(self.q_target, NNEnsembleQFunction):
            next_q = torch.min(next_q, dim=-1)[0]

        log_prob = pi.log_prob(next_action)
        next_v = next_q - self.eta().detach() * log_prob
        not_done = 1.0 - done
        target_q = self.reward_transformer(reward) + self.gamma * next_v * not_done
        return target_q

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
            q_target = self.get_q_target(reward, next_state, done)
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


class MBSoftActorCritic(SoftActorCritic):
    """Model Based Soft-Actor Critic."""

    def __init__(
        self,
        policy,
        q_function,
        dynamical_model,
        reward_model,
        criterion,
        gamma,
        eta,
        regularization=False,
        reward_transformer=RewardTransformer(),
        termination=None,
        num_steps=1,
        num_samples=15,
    ):
        super().__init__(
            policy=policy,
            q_function=q_function,
            criterion=criterion,
            eta=eta,
            regularization=regularization,
            gamma=gamma,
            reward_transformer=reward_transformer,
        )

        self.dynamical_model = dynamical_model
        self.reward_model = reward_model
        self.termination = termination
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.value_function = IntegrateQValueFunction(self.q_target, self.policy, 1)

    def forward(self, state):
        """Compute the losses."""
        with torch.no_grad():
            mc_return, trajectory = mb_return(
                state=state,
                dynamical_model=self.dynamical_model,
                policy=self.policy,
                reward_model=self.reward_model,
                num_steps=self.num_steps,
                gamma=self.gamma,
                reward_transformer=self.reward_transformer,
                value_function=self.value_function,
                num_samples=self.num_samples,
                termination=self.termination,
            )
            next_state = trajectory[-1].next_state
            not_done = 1 - trajectory[-1].done
            pi = tensor_to_distribution(self.policy(next_state), tanh=True)
            next_action = pi.sample()
            log_prob = pi.log_prob(next_action)
            entropy = self.eta().detach() * log_prob * not_done
            target_q = mc_return - self.gamma ** self.num_steps * entropy

        # Critic Loss
        critic_loss, td_error = self.critic_loss(
            trajectory[0].state,
            trajectory[0].action / self.policy.action_scale,
            target_q,
        )

        # Actor loss
        policy_loss, eta_loss = self.actor_loss(state)
        self._info = {"eta": self.eta().detach().item()}

        combined_loss = policy_loss + critic_loss + eta_loss
        return SACLoss(
            loss=combined_loss,
            policy_loss=policy_loss,
            critic_loss=critic_loss,
            eta_loss=eta_loss,
            td_error=td_error,
        )
