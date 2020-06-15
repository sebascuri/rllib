"""Random policy implementation."""

from rllib.util.neural_networks import get_batch_size

from .abstract_policy import AbstractPolicy


class RandomPolicy(AbstractPolicy):
    """Random Policy implementation of AbstractPolicy base class.

    This policy will always return a centered distribution with a unit scaling.

    """

    def __init__(
        self, dim_state, dim_action, num_states=-1, num_actions=-1, action_scale=1.0
    ):
        super().__init__(
            dim_state, dim_action, num_states, num_actions, action_scale=action_scale
        )

    def forward(self, state, **kwargs):
        """Get distribution over actions."""
        batch_size = get_batch_size(state)
        if batch_size:
            return self.random(batch_size, **kwargs)
        else:
            return self.random(**kwargs)
