"""Implementation of OnER Algorithms."""
from rllib.agent.on_policy.advantage_actor_critic_agent import A2CAgent

from .off_policy_agent import OffPolicyAgent


class ISERAgent(OffPolicyAgent):
    """Implementation of an Importance Sampling Experience Replay agent.

    Get a Base On-Policy Agent and expand it with an experience replay.

    Notes
    -----
    The algorithm must implement the importance sampling estimator.

    Parameters
    ----------
    base_agent: OnPolicyAgent.
    memory: ExperienceReplay

    References
    ----------
    Wang, Z., et al. (2016).
    Sample efficient actor-critic with experience replay. ICLR.
    """

    def __init__(self, base_agent, memory, *args, **kwargs):
        super().__init__(memory=memory, *args, **kwargs)
        self.algorithm = base_agent.algorithm
        self.optimizer = base_agent.optimizer
        self.policy = base_agent.policy

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See AbstractAgent.default."""
        return super().default(
            environment, base_agent=A2CAgent.default(environment), *args, **kwargs
        )
