"""Implementation of REINFORCE Algorithms."""
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

    def __init__(self, critic=None, *args, **kwargs):
        super().__init__(algorithm_=REINFORCE, critic=critic, *args, **kwargs)

    @classmethod
    def default(cls, environment, critic=None, *args, **kwargs):
        """See `AbstractAgent.default'."""
        if critic is None:
            critic = NNValueFunction.default(environment)
        return super().default(environment, critic=critic, *args, **kwargs)
