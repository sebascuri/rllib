"""Implementation of a fixed policy agent."""

from rllib.policy import RandomPolicy

from .abstract_agent import AbstractAgent


class FixedPolicyAgent(AbstractAgent):
    """Agent that interacts with an environment using a fixed policy."""

    def __init__(self, policy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy

    @classmethod
    def default(cls, environment, policy=None, *args, **kwargs):
        """See `AbstractAgent.default'."""
        if policy is None:
            policy = RandomPolicy.default(environment, *args, **kwargs)
        return super().default(environment=environment, policy=policy, *args, **kwargs)
