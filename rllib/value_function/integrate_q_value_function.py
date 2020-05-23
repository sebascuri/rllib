"""Value function that is computed by integrating a q-function."""
import torch

from .abstract_value_function import AbstractValueFunction
from .nn_ensemble_value_function import NNEnsembleQFunction
from rllib.util.utilities import integrate, tensor_to_distribution


class IntegrateQValueFunction(AbstractValueFunction):
    """Value function that arises from integrating a q function with a policy.

    Parameters
    ----------
    q_function: AbstractQFunction
        q _function.
    policy: AbstractPolicy
        q _function.
    num_samples: int, optional (default=15).
        Number of states in discrete environments.
    """

    def __init__(self, q_function, policy, num_samples=15, dist_params=None):
        super().__init__(q_function.dim_state, num_states=q_function.num_states,
                         tau=q_function.tau)
        if not isinstance(q_function, NNEnsembleQFunction):
            q_function = NNEnsembleQFunction.from_q_function(q_function, num_heads=1)
        self.q_function = q_function
        self.policy = policy
        self.num_samples = num_samples
        self.dist_params = dict() if dist_params is None else dist_params

    def forward(self, state):
        """Get value of the value-function at a given state."""
        final_v = []
        pi = tensor_to_distribution(self.policy(state), **self.dist_params)
        for i in range(self.q_function.num_heads):
            final_v.append(integrate(
                lambda a: self.q_function(state, a)[i], pi,
                num_samples=self.num_samples)
            )
        final_v = torch.min(*final_v)
        return final_v
