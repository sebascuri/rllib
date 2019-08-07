"""Random policy implementation."""


from .abstract_policy import AbstractPolicy


__all__ = ['RandomPolicy']


class RandomPolicy(AbstractPolicy):
    """Random Policy implementation of AbstractPolicy base class.

    This policy will always return a centered distribution with a scaling given by the
    temperature parameter.

    """

    def __init__(self, dim_state, dim_action, num_states=None, num_actions=None,
                 temperature=1.0):
        super().__init__(dim_state, dim_action, num_states, num_actions, temperature)

    def __call__(self, _):
        return self.random()

    @property
    def parameters(self):
        return None

    @parameters.setter
    def parameters(self, new_params):
        pass
