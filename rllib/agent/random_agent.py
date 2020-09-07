"""Implementation of a random agent."""

from rllib.policy import RandomPolicy

from .abstract_agent import AbstractAgent


class RandomAgent(AbstractAgent):
    """Agent that interacts randomly in an environment."""

    def __init__(
        self, dim_state, dim_action, num_states=-1, num_actions=-1, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.policy = RandomPolicy(
            dim_state, dim_action, num_states=num_states, num_actions=num_actions
        )

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See `AbstractAgent.default'."""
        return super().default(
            environment=environment,
            dim_state=environment.dim_state,
            dim_action=environment.dim_action,
            num_states=environment.num_states,
            num_actions=environment.num_actions,
            *args,
            **kwargs,
        )

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        super().end_episode()
