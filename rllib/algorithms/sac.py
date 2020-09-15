"""Soft Actor-Critic Algorithm."""

import torch

from rllib.dataset.datatypes import Loss
from rllib.util.neural_networks import DisableGradient
from rllib.util.utilities import tensor_to_distribution
from rllib.value_function import NNEnsembleQFunction

from .abstract_algorithm import AbstractAlgorithm


class SoftActorCritic(AbstractAlgorithm):
    r"""Implementation of Soft Actor-Critic algorithm.

    SAC is an off-policy policy gradient algorithm.

    epsilon: Learned temperature as a constraint.
    eta: Fixed regularization.

    References
    ----------
    Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018).
    Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a
    Stochastic Actor. ICML.

    Haarnoja, T., Zhou, A., ... & Levine, S. (2018).
    Soft actor-critic algorithms and applications. arXiv.
    """

    def __init__(self, eta=0.2, regularization=False, *args, **kwargs):
        super().__init__(
            eta=eta, entropy_regularization=regularization, *args, **kwargs
        )
        assert (
            len(self.policy.dim_action) == 1
        ), "Only Nx1 continuous actions implemented."

    def post_init(self):
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
        actor_loss = -q_val

        return Loss(policy_loss=actor_loss).reduce(self.criterion.reduction)

    def get_value_target(self, observation):
        """Get the target of the q function."""
        # Target Q-values
        pi = tensor_to_distribution(
            self.policy(observation.next_state), **self.policy.dist_params
        )
        next_action = self.policy.action_scale * pi.sample()
        next_v = self.critic_target(observation.next_state, next_action)
        if isinstance(self.critic_target, NNEnsembleQFunction):
            next_v = torch.min(next_v, dim=-1)[0]

        next_v = next_v * (1.0 - observation.done)
        aux = self.get_reward(observation) + self.gamma * next_v
        return aux
