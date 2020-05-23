"""Soft Actor-Critic Algorithm."""

import torch

from rllib.util.utilities import tensor_to_distribution, RewardTransformer
from rllib.util.neural_networks import deep_copy_module, disable_gradient, \
    update_parameters
from rllib.util.value_estimation import mb_return
from .abstract_algorithm import AbstractAlgorithm, ACLoss, TDLoss
from rllib.value_function.integrate_q_value_function import IntegrateQValueFunction


class SoftActorCritic(AbstractAlgorithm):
    r"""Implementation of Soft Actor-Critic algorithm.

    SAC is an off-policy policy gradient algorithm.
    """

    def __init__(self, policy, q_function, criterion, temperature, gamma,
                 reward_transformer=RewardTransformer):
        super().__init__()
        # Actor
        self.policy = policy
        self.policy_target = deep_copy_module(policy)

        # Critic
        self.q_function = q_function
        self.q_target = deep_copy_module(q_function)

        self.temperature = temperature

        self.reward_transformer = reward_transformer

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
        with torch.no_grad():
            pi = tensor_to_distribution(self.policy(next_state), tanh=True,
                                        action_scale=self.policy.action_scale)
            next_action = pi.sample()
            next_q = self.q_target(next_state, next_action)
            if type(next_q) is list:
                next_q = torch.min(*next_q)

            next_v = (next_q - self.temperature * pi.log_prob(next_action)) * (1 - done)
            target_q = self.reward_transformer(reward) + self.gamma * next_v
        return target_q

    def forward(self, state, action, reward, next_state, done):
        """Compute the losses."""
        # Critic Loss
        with torch.no_grad():
            q_target = self.get_q_target(reward, next_state, done)
        critic_loss, td_error = self.critic_loss(state, action, q_target)

        # Actor loss
        actor_loss = self.actor_loss(state)
        return ACLoss(loss=critic_loss + actor_loss,
                      actor_loss=actor_loss,
                      critic_loss=critic_loss, td_error=td_error)

    def update(self):
        """Update the baseline network."""
        update_parameters(self.q_target, self.q_function, tau=self.q_function.tau)
        update_parameters(self.policy_target, self.policy, tau=self.policy.tau)


class MBSoftActorCritic(SoftActorCritic):
    """Model Based Soft-Actor Critic."""

    def __init__(self, policy, q_function, dynamical_model, reward_model, criterion,
                 temperature, gamma, reward_transformer=RewardTransformer(),
                 termination=None, num_steps=1, num_samples=15):
        super().__init__(policy=policy, q_function=q_function, criterion=criterion,
                         temperature=temperature, gamma=gamma,
                         reward_transformer=reward_transformer)

        self.dynamical_model = dynamical_model
        self.reward_model = reward_model
        self.termination = termination
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.value_function = IntegrateQValueFunction(
            self.q_target, self.policy_target, self.num_samples,
            dist_params={'tanh': True, 'action_scale': self.policy.action_scale}
        )

    def forward(self, state):
        """Compute the losses."""
        with torch.no_grad():
            mc_return, trajectory = mb_return(
                state=state, dynamical_model=self.dynamical_model,
                policy=self.policy_target,
                reward_model=self.reward_model,
                num_steps=self.num_steps,
                gamma=self.gamma,
                reward_transformer=self.reward_transformer,
                value_function=self.value_function,
                num_samples=self.num_samples,
                termination=self.termination
            )

        critic_loss, td_error = self.critic_loss(
            trajectory[0].state, trajectory[0].action, mc_return)

        # Actor loss
        actor_loss = self.actor_loss(state)
        return ACLoss(actor_loss + critic_loss, actor_loss, critic_loss, td_error)
