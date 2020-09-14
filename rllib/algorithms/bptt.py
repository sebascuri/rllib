"""Back-Propagation Through Time Algorithm."""
from rllib.dataset.datatypes import Loss
from rllib.util.neural_networks.utilities import DisableGradient
from rllib.util.value_estimation import mc_return

from .abstract_algorithm import AbstractAlgorithm
from .abstract_mb_algorithm import AbstractMBAlgorithm


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

    num_samples: int

    def __init__(self, *args, **kwargs):
        AbstractAlgorithm.__init__(self, *args, **kwargs)
        AbstractMBAlgorithm.__init__(self, *args, **kwargs)

    def actor_loss(self, observation):
        """Use the model to compute the gradient loss."""
        sim_observation = self.simulate(observation.state, self.policy)
        with DisableGradient(self.value_target):
            v = mc_return(
                sim_observation,
                gamma=self.gamma,
                value_function=self.value_target,
                reward_transformer=self.reward_transformer,
                reduction="min",
            )
        bptt_loss = Loss(policy_loss=-v).reduce(self.criterion.reduction)
        return bptt_loss
