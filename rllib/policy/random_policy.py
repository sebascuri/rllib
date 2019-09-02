"""Random policy implementation."""


from .abstract_policy import AbstractPolicy
from rllib.util.neural_networks import get_batch_size


__all__ = ['RandomPolicy']


class RandomPolicy(AbstractPolicy):
    """Random Policy implementation of AbstractPolicy base class.

    This policy will always return a centered distribution with a scaling given by the
    temperature parameter.

    """

    def __init__(self, dim_state, dim_action, num_states=None, num_actions=None,
                 temperature=1.0):
        super().__init__(dim_state, dim_action, num_states, num_actions, temperature)

    def __call__(self, states):
        """Get distribution over actions."""
        batch_size = get_batch_size(states)
        if batch_size:
            return self.random(batch_size)
        else:
            return self.random()

    @property
    def parameters(self):
        """See `AbstractPolicy.parameters'."""
        return None

    @parameters.setter
    def parameters(self, new_params):
        """See `AbstractPolicy.parameters.setter'."""
        pass
