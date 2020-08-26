"""Random policy implementation."""

from rllib.util.neural_networks import get_batch_size

from .abstract_policy import AbstractPolicy


class RandomPolicy(AbstractPolicy):
    """Random Policy implementation of AbstractPolicy base class.

    This policy will always return a centered distribution with a unit scaling.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state):
        """Get distribution over actions."""
        batch_size = get_batch_size(state, self.dim_state)
        if batch_size:
            return self.random(batch_size)
        else:
            return self.random()
