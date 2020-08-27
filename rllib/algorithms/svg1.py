"""ModelBasedAlgorithm."""
import torch

from rllib.dataset.datatypes import Loss
from rllib.util.neural_networks.utilities import DisableGradient
from rllib.util.utilities import tensor_to_distribution
from rllib.value_function import NNEnsembleValueFunction

from .bptt import BPTT


class SVG1(BPTT):
    """SVG-ER Algorithm.

    References
    ----------
    Heess, N., Wayne, G., Silver, D., Lillicrap, T., Erez, T., & Tassa, Y. (2015).
    Learning continuous control policies by stochastic value gradients. NeuRIPS.
    """

    def actor_loss(self, observation):
        """Use the model to compute the gradient loss."""
        state, action = observation.state[..., 0, :], observation.action[..., 0, :]
        next_state, done = observation.next_state[..., 0, :], observation.done[..., 0]

        # Infer eta.
        action_mean, action_chol = self.policy(state)
        with torch.no_grad():
            eta = torch.inverse(action_chol) @ ((action - action_mean).unsqueeze(-1))

        # Compute off-policy weight.
        pi = tensor_to_distribution((action_mean, action_chol))
        with torch.no_grad():
            log_p = pi.log_prob(action)
            weight = torch.exp(log_p - observation.log_prob_action[..., 0])

        with DisableGradient(self.dynamical_model, self.reward_model, self.critic):
            # Compute re-parameterized policy sample.
            action = (action_mean + (action_chol @ eta).squeeze(-1)).clamp(-1, 1)

            # Infer xi.
            ns_mean, ns_chol = self.dynamical_model(state, action)
            with torch.no_grad():
                xi = torch.inverse(ns_chol) @ ((next_state - ns_mean).unsqueeze(-1))
            # Compute re-parameterized next-state sample.
            ns = ns_mean + (ns_chol @ xi).squeeze(-1)

            # Compute reward.
            r = tensor_to_distribution(self.reward_model(state, action, ns)).rsample()

            next_v = self.critic_target(ns) * (1 - done)
            if isinstance(self.critic, NNEnsembleValueFunction):
                next_v = next_v[..., 0]

            v = r + self.gamma * next_v

        return Loss(policy_loss=-(weight * v).sum())
