"""Policy derived from an optimistic policy.."""

from .abstract_policy import AbstractPolicy


class DerivedPolicy(AbstractPolicy):
    """Policy derived from an optimistic policy.

    It gets the first `dim_action' components of the base_policy.
    """

    def __init__(self, base_policy: AbstractPolicy, dim_action: int):
        super().__init__(
            dim_state=base_policy.dim_state,
            dim_action=dim_action,
            num_states=base_policy.num_states, num_actions=base_policy.num_actions,
            action_scale=base_policy.action_scale[:dim_action],
            tau=base_policy.tau, deterministic=base_policy.deterministic)
        self.base_policy = base_policy

    def forward(self, state):
        """Compute the derived policy."""
        mean, scale = self.base_policy(state)
        mean = mean[..., :self.dim_action]
        scale = scale[..., :self.dim_action, :self.dim_action]
        return mean, scale

    def reset(self, **kwargs):
        """Reset policy parameters (for example internal states).

        Parameters
        ----------
        kwargs: dict.
            Dictionary with exogenous parameters such as goals.
        """
        self.base_policy.reset(**kwargs)

    def update(self):
        """Update policy parameters."""
        self.base_policy.update()
