"""Back-Propagation Through Time Algorithm."""

from rllib.value_function.model_based_q_function import ModelBasedQFunction

from .abstract_algorithm import AbstractAlgorithm


class BPTT(AbstractAlgorithm):
    """Back-Propagation Through Time Algorithm.

    References
    ----------
    Deisenroth, M., & Rasmussen, C. E. (2011).
    PILCO: A model-based and data-efficient approach to policy search. ICML.


    Clavera, I., Fu, V., & Abbeel, P. (2020).
    Model-Augmented Actor-Critic: Backpropagating through Paths. ICLR.
    """

    def __init__(
        self,
        dynamical_model,
        reward_model,
        num_steps=1,
        lambda_=1.0,
        termination_model=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if num_steps > 0:
            self.pathwise_loss.critic = ModelBasedQFunction(
                dynamical_model=dynamical_model,
                reward_model=reward_model,
                termination_model=termination_model,
                num_samples=self.num_samples,
                num_steps=num_steps,
                policy=self.policy,
                value_function=self.value_function,
                gamma=self.gamma,
                lambda_=lambda_,
                reward_transformer=self.reward_transformer,
                entropy_regularization=self.entropy_loss.eta.item(),
            )

    def actor_loss(self, observation):
        """Use the model to compute the gradient loss."""
        return self.pathwise_loss(observation).reduce(self.criterion.reduction)

    def update(self):
        """Update algorithm parameters."""
        super().update()
        self.pathwise_loss.critic.entropy_regularization = self.entropy_loss.eta.item()
