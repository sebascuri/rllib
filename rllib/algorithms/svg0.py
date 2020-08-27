"""SVG-0 Algorithm is DPG with stochastic re-parameterized policies."""
import torch

from rllib.dataset.datatypes import Loss
from rllib.util.neural_networks.utilities import DisableGradient
from rllib.util.utilities import get_entropy_and_logp, tensor_to_distribution
from rllib.value_function import NNEnsembleQFunction

from .dpg import DPG


class SVG0(DPG):
    """Implementation of SVG-0 Algorithm."""

    def __init__(self, *args, **kwargs):
        super().__init__(policy_noise=0, noise_clip=0, *args, **kwargs)

    def actor_loss(self, observation) -> Loss:
        """Compute Actor loss."""
        action_mean, action_chol = self.policy(observation.state[..., 0, :])
        # Infer eta.
        with torch.no_grad():
            delta = observation.action[..., 0, :] - action_mean
            eta = torch.inverse(action_chol) @ delta.unsqueeze(-1)

        # Compute entropy and log_probability.
        pi = tensor_to_distribution((action_mean, action_chol))
        entropy, log_p = get_entropy_and_logp(
            pi=pi, action=observation.action[..., 0, :]
        )

        # Compute off-policy weight.
        with torch.no_grad():
            weight = torch.exp(log_p - observation.log_prob_action[..., 0])

        # Compute re-parameterized policy sample.
        action = (action_mean + (action_chol @ eta).squeeze(-1)).clamp(-1, 1)

        # Propagate gradient.
        with DisableGradient(self.critic):
            q = self.critic(observation.state[..., 0, :], action)
            if isinstance(self.critic, NNEnsembleQFunction):
                q = q[..., 0]

        return Loss(
            policy_loss=-weight * q,
            regularization_loss=-self.entropy_regularization * entropy,
        )
