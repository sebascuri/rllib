"""Policy derived from an optimistic policy.."""
from .abstract_policy import AbstractPolicy
from .nn_policy import NNPolicy


class DerivedPolicy(AbstractPolicy):
    """Policy derived from an optimistic policy.

    It gets the first `dim_action' components of the base_policy.
    """

    def __init__(self, base_policy: AbstractPolicy, dim_action, *args, **kwargs):
        if base_policy.discrete_action or base_policy.discrete_state:
            raise NotImplementedError
        assert len(dim_action) == 1, "Only n x 1 actions allowed."
        super().__init__(
            dim_state=base_policy.dim_state,
            dim_action=dim_action,
            num_states=base_policy.num_states,
            num_actions=base_policy.num_actions,
            action_scale=base_policy.action_scale[: dim_action[0]],
            tau=base_policy.tau,
            deterministic=base_policy.deterministic,
            goal=base_policy.goal,
            dist_params=base_policy.dist_params,
        )
        self.base_policy = base_policy

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See AbstractPolicy.default()."""
        kwargs.pop("dim_action", None)
        return super().default(
            environment=environment,
            base_policy=NNPolicy.default(environment),
            dim_action=environment.dim_action,
            *args,
            **kwargs,
        )

    def forward(self, state, **kwargs):
        """Compute the derived policy."""
        mean, scale = self.base_policy(state, **kwargs)
        mean = mean[..., : self.dim_action[0]]
        scale = scale[..., : self.dim_action[0], : self.dim_action[0]]
        return mean, scale

    def reset(self):
        """Reset policy parameters."""
        self.base_policy.reset()

    def update(self):
        """Update policy parameters."""
        self.base_policy.update()

    def set_goal(self, goal):
        """Update policy parameters."""
        self.base_policy.set_goal(goal)
