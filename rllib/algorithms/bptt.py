"""Back-Propagation Through Time Algorithm."""
from rllib.dataset.datatypes import Loss
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.util.neural_networks.utilities import DisableGradient
from rllib.util.value_estimation import mc_return

from .abstract_algorithm import AbstractAlgorithm
from .abstract_mb_algorithm import AbstractMBAlgorithm
from .kl_loss import KLLoss


class BPTT(AbstractAlgorithm, AbstractMBAlgorithm):
    """Back-Propagation Through Time Algorithm.

    References
    ----------
    Deisenroth, M., & Rasmussen, C. E. (2011).
    PILCO: A model-based and data-efficient approach to policy search. ICML.

    Parmas, P., Rasmussen, C. E., Peters, J., & Doya, K. (2018).
    PIPPS: Flexible model-based policy search robust to the curse of chaos. ICML.

    Clavera, I., Fu, V., & Abbeel, P. (2020).
    Model-Augmented Actor-Critic: Backpropagating through Paths. ICLR.
    """

    def __init__(
        self, epsilon_mean=0.0, epsilon_var=0.0, regularization=True, *args, **kwargs
    ):
        AbstractAlgorithm.__init__(self, *args, **kwargs)
        AbstractMBAlgorithm.__init__(self, *args, **kwargs)
        self.kl_loss = KLLoss(epsilon_mean, epsilon_var, regularization)

    def actor_loss(self, observation):
        """Use the model to compute the gradient loss."""
        trajectory = self.simulate(observation.state, self.policy)
        observation = stack_list_of_tuples(trajectory, dim=-2)
        with DisableGradient(self.value_target):
            v = mc_return(
                observation,
                gamma=self.gamma,
                value_function=self.value_target,
                reward_transformer=self.reward_transformer,
                reduction="min",
            )
        state, action = observation.state, observation.action
        _, _, kl_mean, kl_var, entropy = self.get_log_p_kl_entropy(state, action)
        bptt_loss = Loss(
            policy_loss=-v.mean(),
            regularization_loss=-self.entropy_regularization * entropy,
        )
        kl_loss = self.kl_loss(kl_mean, kl_var)
        return bptt_loss + kl_loss
