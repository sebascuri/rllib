"""ModelBasedAlgorithm."""
import torch

from rllib.dataset.datatypes import Loss
from rllib.util.neural_networks.utilities import DisableGradient
from rllib.util.utilities import tensor_to_distribution
from rllib.value_function import AbstractValueFunction

from .abstract_algorithm import AbstractAlgorithm
from .abstract_mb_algorithm import AbstractMBAlgorithm


class SVG(AbstractAlgorithm, AbstractMBAlgorithm):
    """SVG Algorithm.

    References
    ----------
    Heess, N., Wayne, G., Silver, D., Lillicrap, T., Erez, T., & Tassa, Y. (2015).
    Learning continuous control policies by stochastic value gradients. NeuRIPS.
    """

    def __init__(self, *args, **kwargs):
        AbstractAlgorithm.__init__(*args, **kwargs)
        AbstractMBAlgorithm.__init__(self, *args, **kwargs)
        assert isinstance(self.critic, AbstractValueFunction)

    def actor_loss(self, observation):
        """Use the model to compute the gradient loss."""
        state, action = observation.state, observation.action

        log_p_old = observation.log_prob_action

        pi = tensor_to_distribution(self.policy(state))
        with torch.no_grad():
            log_p = pi.log_prob(action)
            weight = torch.exp(log_p - log_p_old)

        with DisableGradient(self.dynamical_model, self.reward_model, self.critic):
            action_ = pi.rsample()
            r = tensor_to_distribution(self.reward_model(state, action_)).rsample()

            ns_mean, ns_chol = self.dynamical_model(state, action_)

            # Infer xi.
            with torch.no_grad():
                xi = torch.inverse(ns_chol) @ (observation.next_state - ns_mean)
            ns = ns_mean + xi @ ns_chol

            next_v = self.critic_target(ns) * (1 - observation.done)
            v = weight * (r + self.gamma * next_v)
        return Loss(policy_loss=v.sum())
