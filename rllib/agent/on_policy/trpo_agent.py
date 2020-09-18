"""Implementation of TRPO Algorithm."""
from torch.nn.modules import loss

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
        policy,
        critic,
        regularization=False,
        epsilon_mean=0.01,
        epsilon_var=None,
        lambda_=0.95,
        criterion=loss.MSELoss,
        *args,
        **kwargs,
    ):
        super().__init__(policy=policy, critic=critic, *args, **kwargs)

        self.algorithm = TRPO(
            critic=critic,
            policy=policy,
            regularization=regularization,
            epsilon_mean=epsilon_mean,
            epsilon_var=epsilon_var,
            criterion=criterion(reduction="mean"),
            lambda_=lambda_,
            *args,
            **kwargs,
        )
        # Over-write optimizer.
        self.optimizer = type(self.optimizer)(
            [p for n, p in self.algorithm.named_parameters() if "target" not in n],
            **self.optimizer.defaults,
        )

        self.policy = self.algorithm.policy

    @classmethod
    def default(cls, environment, num_iter=80, *args, **kwargs):
        """See `AbstractAgent.default'."""
        return super().default(
            environment,
            critic=NNValueFunction.default(environment),
            num_iter=num_iter,
            *args,
            **kwargs,
        )
