"""Interface for Q-Function derived policies."""

from abc import ABCMeta

from rllib.util.parameter_decay import Constant, ParameterDecay
from rllib.value_function.nn_value_function import NNQFunction

from ..abstract_policy import AbstractPolicy


class AbstractQFunctionPolicy(AbstractPolicy, metaclass=ABCMeta):
    """Interface for policies to control an environment.

    Parameters
    ----------
    q_function: q_function to derive policy from.
    param: policy parameter.

    """

    def __init__(self, q_function, param, *args, **kwargs):
        if not q_function.discrete_action:
            raise NotImplementedError
        super().__init__(
            dim_state=q_function.dim_state,
            dim_action=q_function.dim_action,
            num_states=q_function.num_states,
            num_actions=q_function.num_actions,
        )
        self.q_function = q_function
        if not isinstance(param, ParameterDecay):
            param = Constant(param)
        self.param = param

    def update(self):
        """Update policy parameters."""
        self.param.update()

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See AbstractPolicy.default."""
        return super().default(
            environment=environment,
            q_function=NNQFunction.default(environment, *args, **kwargs),
            param=0.1,
        )
