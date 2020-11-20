"""Implementation of REINFORCE Algorithms."""
from torch.nn.modules import loss

from rllib.algorithms.reinforce import REINFORCE
from rllib.value_function import NNValueFunction

from .actor_critic_agent import ActorCriticAgent


class REINFORCEAgent(ActorCriticAgent):
    """Implementation of the REINFORCE algorithm.

    The REINFORCE algorithm computes the policy gradient using MC
    approximation for the returns (sum of discounted rewards).

    References
    ----------
    Williams, Ronald J. "Simple statistical gradient-following algorithms for
    connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.
    """

    def __init__(self, policy, critic=None, criterion=loss.MSELoss, *args, **kwargs):
        super().__init__(policy=policy, critic=critic, *args, **kwargs)
        self.algorithm = REINFORCE(
            policy=policy,
            baseline=critic,
            criterion=criterion(reduction="mean"),
            *args,
            **kwargs,
        )
        self.policy = self.algorithm.policy

    @classmethod
    def default(cls, environment, critic=None, *args, **kwargs):
        """See `AbstractAgent.default'."""
        if critic is None:
            critic = NNValueFunction.default(environment)
        return super().default(environment, critic=critic, *args, **kwargs)
