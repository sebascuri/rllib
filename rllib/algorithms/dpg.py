"""Deterministic Policy Gradient Algorithm."""

import torch

from rllib.dataset.datatypes import Loss
from rllib.util.neural_networks.utilities import DisableGradient
from rllib.util.utilities import tensor_to_distribution
from rllib.value_function import NNEnsembleQFunction

from .abstract_algorithm import AbstractAlgorithm


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

    def __init__(self, policy_noise=0.0, noise_clip=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_target.dist_params.update(
            add_noise=True, policy_noise=policy_noise, noise_clip=noise_clip
        )
        self.value_target.policy = self.policy_target

    def actor_loss(self, observation):
        """Get Actor Loss."""
        state = observation.state
        action = tensor_to_distribution(
            self.policy(state), **self.policy.dist_params
        ).mean.clamp(-1, 1)
        with DisableGradient(self.critic_target):
            q = self.critic_target(state, action)
            if isinstance(self.critic_target, NNEnsembleQFunction):
                q = q[..., 0]
        return Loss(policy_loss=-q)

    def get_value_target(self, observation):
        """Get q function target."""
        next_v = self.value_target(observation.next_state)
        if isinstance(self.critic_target, NNEnsembleQFunction):
            next_v = torch.min(next_v, dim=-1)[0]
        next_v = next_v * (1 - observation.done)
        return self.reward_transformer(observation.reward) + self.gamma * next_v
