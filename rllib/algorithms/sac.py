"""Soft Actor-Critic Algorithm."""

import torch

from rllib.util.utilities import tensor_to_distribution, RewardTransformer
from rllib.util.neural_networks import deep_copy_module, disable_gradient, \
    update_parameters
from rllib.util.value_estimation import mb_return
from .abstract_algorithm import AbstractAlgorithm, SACLoss, TDLoss
from rllib.value_function.integrate_q_value_function import IntegrateQValueFunction
from rllib.util.parameter_decay import ParameterDecay, Learnable, Constant


class SoftActorCritic(AbstractAlgorithm):
    r"""Implementation of Soft Actor-Critic algorithm.

    SAC is an off-policy policy gradient algorithm.

    epsilon: Learned temperature as a constraint.
    eta: Fixed regularization.
    """

    def __init__(self, policy, q_function, criterion, gamma, epsilon=None, eta=None,
                 reward_transformer=RewardTransformer()):
        super().__init__()
        # Actor
        self.policy = policy
        self.target_entropy = -policy.dim_action

        # Critic
        self.q_function = q_function
        self.q_target = deep_copy_module(q_function)

        assert (epsilon is not None) ^ (eta is not None), "XOR(eps, eta)."
        if eta is not None:  # Regularization: \eta KL(\pi || Uniform)
            if not isinstance(eta, ParameterDecay):
                eta = Constant(eta)
            self.eta = eta
        else:  # Trust-Region: || KL(\pi || Uniform)|| < \epsilon
            self.eta = Learnable(epsilon, positive=True)

        self.reward_transformer = reward_transformer

        self.criterion = criterion
        self.gamma = gamma
        self.dist_params = {'tanh': True, 'action_scale': self.policy.action_scale}

    def actor_loss(self, state):
        """Get Actor Loss."""
        pi = tensor_to_distribution(self.policy(state, normalized=True), tanh=True)
        action = pi.rsample()  # re-parametrization trick.

        with disable_gradient(self.q_target):
            q_val = self.q_target(state, action)
            if type(q_val) is list:
                q_val = torch.min(*q_val)

        log_prob = pi.log_prob(action)
        eta_loss = (self.eta() * (-log_prob - self.target_entropy).detach())
        actor_loss = self.eta().detach() * log_prob - q_val

        if self.criterion.reduction == 'mean':
            eta_loss, actor_loss = eta_loss.mean(), actor_loss.mean()
        elif self.criterion.reduction == 'sum':
            eta_loss, actor_loss = eta_loss.sum(), actor_loss.sum()

        return actor_loss, eta_loss

    def critic_loss(self, state, action, q_target):
        """Get Critic Loss and td-error."""
        pred_q = self.q_function(state, action)
        if type(pred_q) is not list:
            pred_q = [pred_q]

        critic_loss = self.criterion(pred_q[0], q_target)
        td_error = (pred_q[0] - q_target).detach()
        for q in pred_q[1:]:
            critic_loss += self.criterion(q, q_target)
            td_error += (q - q_target).detach()

        return TDLoss(critic_loss, td_error)

    def get_q_target(self, reward, next_state, done):
        """Get the target of the q function."""
        # Target Q-values
        pi = tensor_to_distribution(self.policy(next_state, normalized=True), tanh=True)
        next_action = pi.sample()
        next_q = self.q_target(next_state, next_action)
        if type(next_q) is list:
            next_q = torch.min(*next_q)

        log_prob = pi.log_prob(next_action)
        next_v = (next_q - self.eta().detach() * log_prob)
        not_done = 1. - done
        target_q = self.reward_transformer(reward) + self.gamma * next_v * not_done
        return target_q

    def forward(self, state, action, reward, next_state, done):
        """Compute the losses."""
        # Critic Loss
        with torch.no_grad():
            q_target = self.get_q_target(reward, next_state, done)
        critic_loss, td_error = self.critic_loss(
            state, action / self.policy.action_scale, q_target)

        # Actor loss
        policy_loss, eta_loss = self.actor_loss(state)
        self._info = {'eta': self.eta().detach().item()}

        return SACLoss(loss=policy_loss + critic_loss + critic_loss + eta_loss,
                       policy_loss=policy_loss, critic_loss=critic_loss,
                       eta_loss=eta_loss, td_error=td_error)

    def update(self):
        """Update the baseline network."""
        update_parameters(self.q_target, self.q_function, tau=self.q_function.tau)


class MBSoftActorCritic(SoftActorCritic):
    """Model Based Soft-Actor Critic."""

    def __init__(self, policy, q_function, dynamical_model, reward_model, criterion,
                 gamma, epsilon=None, eta=None, reward_transformer=RewardTransformer(),
                 termination=None, num_steps=1, num_samples=15):
        super().__init__(policy=policy, q_function=q_function, criterion=criterion,
                         epsilon=epsilon, eta=eta, gamma=gamma,
                         reward_transformer=reward_transformer)

        self.dynamical_model = dynamical_model
        self.reward_model = reward_model
        self.termination = termination
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.value_function = IntegrateQValueFunction(
            self.q_target, self.policy, 1, dist_params=self.dist_params)

    def forward(self, state):
        """Compute the losses."""
        with torch.no_grad():
            mc_return, trajectory = mb_return(
                state=state, dynamical_model=self.dynamical_model,
                policy=self.policy,
                reward_model=self.reward_model,
                num_steps=self.num_steps,
                gamma=self.gamma,
                reward_transformer=self.reward_transformer,
                value_function=self.value_function,
                num_samples=self.num_samples,
                termination=self.termination,
                **self.dist_params
            )
            next_state = trajectory[-1].next_state
            not_done = 1 - trajectory[-1].done
            pi = tensor_to_distribution(self.policy(next_state),
                                        **self.dist_params)
            next_action = pi.sample()
            log_prob = pi.log_prob(next_action)
            entropy = self.eta().detach() * log_prob * not_done
            target_q = mc_return - self.gamma ** self.num_steps * entropy

        # Critic Loss
        critic_loss, td_error = self.critic_loss(
            trajectory[0].state, trajectory[0].action / self.policy.action_scale,
            target_q)

        # Actor loss
        policy_loss, eta_loss = self.actor_loss(state)
        self._info = {'eta': self.eta().detach().item()}

        combined_loss = policy_loss + critic_loss + eta_loss
        return SACLoss(loss=combined_loss, policy_loss=policy_loss,
                       critic_loss=critic_loss, eta_loss=eta_loss, td_error=td_error)
