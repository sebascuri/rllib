"""Deterministic Policy Gradient Algorithm."""

import torch

from rllib.util.neural_networks.utilities import DisableGradient
from rllib.util.utilities import tensor_to_distribution
from rllib.value_function import NNEnsembleQFunction

from .abstract_algorithm import AbstractAlgorithm, Loss


class DPG(AbstractAlgorithm):
    r"""Implementation of DPG algorithm.

    DPG is an off-policy model-free control algorithm.

    The DPG algorithm is an actor-critic algorithm that has a policy that estimates:
    .. math:: a = \pi(s) = \argmax_a Q(s, a)


    Parameters
    ----------
    critic: AbstractQFunction
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

    def __init__(self, policy_noise, noise_clip, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

    def actor_loss(self, observation):
        """Get Actor Loss."""
        state = observation.state
        action = tensor_to_distribution(self.policy(state)).mean.clamp(-1, 1)
        with DisableGradient(self.critic):
            q = self.critic(state, action)
            if isinstance(self.critic, NNEnsembleQFunction):
                q = q[..., 0]
        return Loss(loss=-q, policy_loss=-q)

    def get_value_target(self, observation):
        """Get q function target."""
        next_action = tensor_to_distribution(
            self.policy_target(observation.next_state),
            add_noise=True,
            policy_noise=self.policy_noise,
            noise_clip=self.noise_clip,
        ).sample()
        next_v = self.critic_target(observation.next_state, next_action)
        if isinstance(self.critic_target, NNEnsembleQFunction):
            next_v = torch.min(next_v, dim=-1)[0]
        next_v = next_v * (1 - observation.done)
        return self.reward_transformer(observation.reward) + self.gamma * next_v

    def forward_slow(self, observation):
        """Compute the losses and the td-error."""
        # Critic Loss
        critic_loss = self.critic_loss(observation)

        # Actor loss
        actor_loss = self.actor_loss(observation)

        return Loss(
            loss=actor_loss.policy_loss + critic_loss.critic_loss,
            policy_loss=actor_loss.policy_loss,
            critic_loss=critic_loss.critic_loss,
            td_error=critic_loss.td_error,
        )
