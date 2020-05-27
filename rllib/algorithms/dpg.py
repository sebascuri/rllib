"""Deterministic Policy Gradient Algorithm."""

import torch

from rllib.util.neural_networks.utilities import disable_gradient, deep_copy_module, \
    update_parameters
from rllib.util.utilities import tensor_to_distribution, RewardTransformer
from .abstract_algorithm import AbstractAlgorithm, ACLoss, TDLoss
from rllib.value_function.integrate_q_value_function import IntegrateQValueFunction
from rllib.util.value_estimation import mb_return


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

    def __init__(self, q_function, policy, criterion, gamma, policy_noise, noise_clip,
                 reward_transformer=RewardTransformer()):
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
        self.reward_transformer = reward_transformer

    def actor_loss(self, state):
        """Get Actor Loss."""
        action = tensor_to_distribution(self.policy(state)).mean.clamp(-1, 1)
        with disable_gradient(self.q_function):
            q = self.q_function(state, action)
            if type(q) is list:
                q = q[0]
        return -q

    def get_q_target(self, reward, next_state, done):
        """Get q function target."""
        next_action = tensor_to_distribution(self.policy_target(next_state), **{
            'add_noise': True,
            'action_scale': self.policy.action_scale,
            'policy_noise': self.policy_noise,
            'noise_clip': self.noise_clip}
                                             ).sample()
        next_v = self.q_target(next_state, next_action)
        if type(next_v) is list:
            next_v = torch.min(*next_v)
        q_target = self.reward_transformer(reward) + self.gamma * next_v * (1 - done)
        return q_target

    def critic_loss(self, state, action, q_target):
        """Get Critic Loss and td-error."""
        pred_q = self.q_function(state, action)
        if type(pred_q) is not list:
            pred_q = [pred_q]

        # Target Q-values
        critic_loss = self.criterion(pred_q[0], q_target)
        td_error = pred_q[0].detach() - q_target.detach()
        for q in pred_q[1:]:
            critic_loss += (self.criterion(q, q_target))
            td_error += q.detach() - q_target.detach()

        return TDLoss(critic_loss, td_error)

    def forward(self, state, action, reward, next_state, done):
        """Compute the losses and the td-error."""
        # Critic Loss
        with torch.no_grad():
            q_target = self.get_q_target(reward, next_state, done)
        critic_loss, td_error = self.critic_loss(state, action, q_target)

        # Actor loss
        actor_loss = self.actor_loss(state)

        return ACLoss(
            loss=(actor_loss + critic_loss).squeeze(-1),
            policy_loss=actor_loss.squeeze(-1),
            critic_loss=critic_loss.squeeze(-1),
            td_error=td_error.squeeze(-1))

    def update(self):
        """Update the target network."""
        update_parameters(self.policy_target, self.policy, tau=self.policy.tau)
        update_parameters(self.q_target, self.q_function, tau=self.q_function.tau)


class MBDPG(DPG):
    """Model based implementation of DPG."""

    def __init__(self, policy, q_function, dynamical_model, reward_model, criterion,
                 policy_noise, noise_clip, gamma,
                 reward_transformer=RewardTransformer(),
                 termination=None, num_steps=1, num_samples=15):
        super().__init__(policy=policy, q_function=q_function, criterion=criterion,
                         gamma=gamma, reward_transformer=reward_transformer,
                         policy_noise=policy_noise, noise_clip=noise_clip)
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
            q_target, trajectory = mb_return(
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

        critic_loss, td_error = self.critic_loss(trajectory[0].state,
                                                 trajectory[0].action,
                                                 q_target)

        # Actor loss
        actor_loss = self.actor_loss(state)

        return ACLoss(
            loss=(actor_loss + critic_loss).squeeze(-1),
            policy_loss=actor_loss.squeeze(-1),
            critic_loss=critic_loss.squeeze(-1),
            td_error=td_error.squeeze(-1)
        )
