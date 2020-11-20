"""SVG-0 Algorithm is DPG with stochastic re-parameterized policies."""
import torch

from rllib.dataset.datatypes import Loss
from rllib.util.neural_networks.utilities import DisableGradient
from rllib.value_function import NNEnsembleQFunction

from .dpg import DPG


class SVG0(DPG):
    """Implementation of SVG-0 Algorithm."""

    def __init__(self, *args, **kwargs):
        super().__init__(policy_noise=0, noise_clip=0, *args, **kwargs)

    def actor_loss(self, observation) -> Loss:
        """Compute Actor loss."""
        state, action = observation.state[..., 0, :], observation.action[..., 0, :]
        action_mean, action_chol = self.policy(state)

        # Infer eta.
        with torch.no_grad():
            delta = action / self.policy.action_scale - action_mean
            eta = torch.inverse(action_chol) @ delta.unsqueeze(-1)

        # Compute re-parameterized policy sample.
        action = self.policy.action_scale * (
            action_mean + (action_chol @ eta).squeeze(-1)
        ).clamp(-1.0, 1.0)

        # Propagate gradient.
        with DisableGradient(self.critic):
            q = self.critic(observation.state[..., 0, :], action)
            if isinstance(self.critic, NNEnsembleQFunction):
                q = q[..., 0]

        return Loss(policy_loss=-q).reduce(self.criterion.reduction)
