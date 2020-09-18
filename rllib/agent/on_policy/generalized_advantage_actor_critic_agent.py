"""Implementation of Advantage-Actor Critic Agent."""
from torch.nn.modules import loss

from rllib.algorithms.gaac import GAAC
from rllib.value_function import NNValueFunction

from .actor_critic_agent import ActorCriticAgent


class GAACAgent(ActorCriticAgent):
    """Implementation of the Advantage-Actor Critic.

    TODO: build compatible function approximation.

    References
    ----------
    Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015).
    High-dimensional continuous control using generalized advantage estimation. ICLR.
    """

    def __init__(
        self, policy, critic, lambda_=0.97, criterion=loss.MSELoss, *args, **kwargs
    ):
        super().__init__(policy=policy, critic=critic, *args, **kwargs)
        self.algorithm = GAAC(
            policy=policy,
            critic=critic,
            criterion=criterion(reduction="mean"),
            lambda_=lambda_,
            *args,
            **kwargs,
        )
        self.policy = self.algorithm.policy

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See `AbstractAgent.default'."""
        return super().default(
            environment, critic=NNValueFunction.default(environment), *args, **kwargs
        )
