"""Interface for Q-Function derived policies."""

from abc import abstractmethod
from ..abstract_policy import AbstractPolicy
from rllib.util import ExponentialDecay


class AbstractQFunctionPolicy(AbstractPolicy):
    """Interface for policies to control an environment.

    Parameters
    ----------
    q_function: q_function to derive policy from.
    start: starting parameter.
    end: finishing parameter.
    decay: decay parameter.

    """

    def __init__(self, q_function, start, end=None, decay=None):
        if not q_function.discrete_action:
            raise NotImplementedError
        self.q_function = q_function
        self.param = ExponentialDecay(start, end, decay)
        super().__init__(q_function.dim_state,
                         q_function.dim_action,
                         num_states=q_function.num_states,
                         num_actions=q_function.num_actions)

    @abstractmethod
    def __call__(self, state):
        """Return the action distribution of the policy.

        Parameters
        ----------
        state: tensor

        Returns
        -------
        action: torch.distributions.Distribution

        """
        raise NotImplementedError

    @property
    def parameters(self):
        """Get policy parameters."""
        return self.q_function.parameters

    @parameters.setter
    def parameters(self, new_params):
        """Set policy parameters."""
        self.q_function.parameters = new_params
