"""Policy derived from an optimistic policy.."""

from .abstract_policy import AbstractPolicy


class DerivedPolicy(AbstractPolicy):
    """Policy derived from an optimistic policy.

    It gets the first `dim_action' components of the base_policy.
    """

    def __init__(self, base_policy: AbstractPolicy):
        super().__init__(
            dim_state=base_policy.dim_state,
            dim_action=base_policy.dim_action - base_policy.dim_state,
            num_states=base_policy.num_states, num_actions=base_policy.num_actions,
            tau=base_policy.tau, deterministic=base_policy.deterministic)
        self.base_policy = base_policy

    def forward(self, state):
        """Compute the derived policy."""
        mean, scale = self.base_policy(state)
        mean = mean[..., :-self.num_states]
        scale = scale[..., :-self.num_states, :-self.num_states]
        return mean, scale
