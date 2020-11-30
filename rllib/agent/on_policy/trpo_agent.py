"""Implementation of TRPO Algorithm."""
from rllib.algorithms.trpo import TRPO
from rllib.value_function import NNValueFunction

from .actor_critic_agent import ActorCriticAgent


class TRPOAgent(ActorCriticAgent):
    """Implementation of the TRPO Agent.

    References
    ----------
    Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015).
    Trust region policy optimization. ICML.
    """

    def __init__(
        self,
        kl_regularization=False,
        epsilon_mean=0.01,
        epsilon_var=None,
        lambda_=0.95,
        num_iter=80,
        num_rollouts=4,
        *args,
        **kwargs,
    ):
        super().__init__(
            algorithm_=TRPO,
            kl_regularization=kl_regularization,
            epsilon_mean=epsilon_mean,
            epsilon_var=epsilon_var,
            lambda_=lambda_,
            num_iter=num_iter,
            num_rollouts=num_rollouts,
            *args,
            **kwargs,
        )

    @classmethod
    def default(cls, environment, critic=None, *args, **kwargs):
        """See `AbstractAgent.default'."""
        if critic is None:
            critic = NNValueFunction.default(environment)
        return super().default(environment, critic=critic, *args, **kwargs)
