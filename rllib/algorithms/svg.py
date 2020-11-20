"""ModelBasedAlgorithm."""
import torch

from rllib.dataset.datatypes import Loss
from rllib.model import TransformedModel
from rllib.util.neural_networks.utilities import DisableGradient
from rllib.util.utilities import get_entropy_and_log_p, tensor_to_distribution
from rllib.value_function import NNEnsembleQFunction, NNEnsembleValueFunction

from .abstract_algorithm import AbstractAlgorithm


class SVG(AbstractAlgorithm):
    """Stochastic Value Gradient Algorithm.

    References
    ----------
    Heess, N., Wayne, G., Silver, D., Lillicrap, T., Erez, T., & Tassa, Y. (2015).
    Learning continuous control policies by stochastic value gradients. NeuRIPS.
    """

    def __init__(
        self, dynamical_model, reward_model, termination_model=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not isinstance(dynamical_model, TransformedModel):
            dynamical_model = TransformedModel(dynamical_model, [])
        if not isinstance(reward_model, TransformedModel):
            reward_model = TransformedModel(reward_model, [])
        self.dynamical_model = dynamical_model
        self.reward_model = reward_model
        self.termination_model = termination_model

    def actor_loss(self, observation):
        """Use the model to compute the gradient loss."""
        state, action = observation.state, observation.action
        next_state, done = observation.next_state, observation.done

        # Infer eta.
        action_mean, action_chol = self.policy(state)
        with torch.no_grad():
            eta = torch.inverse(action_chol) @ ((action - action_mean).unsqueeze(-1))

        # Compute entropy and log_probability.
        pi = tensor_to_distribution((action_mean, action_chol))
        _, log_p = get_entropy_and_log_p(pi, action, self.policy.action_scale)

        # Compute off-policy weight.
        with torch.no_grad():
            weight = self.get_ope_weight(state, action, observation.log_prob_action)

        with DisableGradient(
            self.dynamical_model,
            self.reward_model,
            self.termination_model,
            self.critic_target,
        ):
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
            r = r[..., 0]

            next_v = self.value_function(ns)
            if isinstance(self.critic, NNEnsembleValueFunction) or isinstance(
                self.critic, NNEnsembleQFunction
            ):
                next_v = next_v[..., 0]

            v = r + self.gamma * next_v * (1 - done)

        return Loss(policy_loss=-(weight * v)).reduce(self.criterion.reduction)
