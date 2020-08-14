"""Deterministic Policy Gradient Algorithm."""

import torch

from rllib.util.neural_networks.utilities import (
    DisableGradient,
    deep_copy_module,
    update_parameters,
)
from rllib.util.utilities import tensor_to_distribution
from rllib.value_function import NNEnsembleQFunction

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

    def __init__(
        self, q_function, criterion, policy_noise, noise_clip, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        # Critic
        self.q_function = q_function
        self.q_target = deep_copy_module(q_function)
        self.criterion = criterion

        # Actor
        self.policy_target = deep_copy_module(self.policy)

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

    def actor_loss(self, state):
        """Get Actor Loss."""
        action = tensor_to_distribution(self.policy(state)).mean.clamp(-1, 1)
        with DisableGradient(self.q_function):
            q = self.q_function(state, action)
            if isinstance(self.q_function, NNEnsembleQFunction):
                q = q[..., 0]
        return -q

    def get_q_target(self, observation):
        """Get q function target."""
        next_action = tensor_to_distribution(
            self.policy_target(observation.next_state),
            add_noise=True,
            policy_noise=self.policy_noise,
            noise_clip=self.noise_clip,
        ).sample()
        next_v = self.q_target(observation.next_state, next_action)
        if isinstance(self.q_target, NNEnsembleQFunction):
            next_v = torch.min(next_v, dim=-1)[0]
        next_v = next_v * (1 - observation.done)
        return self.reward_transformer(observation.reward) + self.gamma * next_v

    def critic_loss(self, state, action, q_target):
        """Get Critic Loss and td-error."""
        pred_q = self.q_function(state, action)
        if isinstance(self.q_function, NNEnsembleQFunction):
            q_target = q_target.unsqueeze(-1).repeat_interleave(
                self.q_function.num_heads, -1
            )
        # Target Q-values
        critic_loss = self.criterion(pred_q, q_target)
        td_error = pred_q.detach() - q_target.detach()

        if isinstance(self.q_function, NNEnsembleQFunction):
            critic_loss = critic_loss.sum(-1)
            td_error = td_error.sum(-1)

        return TDLoss(critic_loss, td_error)

    def forward(self, observation):
        """Compute the losses and the td-error."""
        state, action, reward, next_state, done, *r = observation
        # Critic Loss
        with torch.no_grad():
            q_target = self.get_q_target(observation)
        critic_loss, td_error = self.critic_loss(state, action, q_target)

        # Actor loss
        actor_loss = self.actor_loss(state)

        return ACLoss(
            loss=(actor_loss + critic_loss).squeeze(-1),
            policy_loss=actor_loss.squeeze(-1),
            critic_loss=critic_loss.squeeze(-1),
            td_error=td_error.squeeze(-1),
        )

    def update(self):
        """Update the target network."""
        update_parameters(self.policy_target, self.policy, tau=self.policy.tau)
        update_parameters(self.q_target, self.q_function, tau=self.q_function.tau)
