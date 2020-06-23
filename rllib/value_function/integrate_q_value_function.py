"""Value function that is computed by integrating a q-function."""

from rllib.util.utilities import integrate, tensor_to_distribution

from .abstract_value_function import AbstractValueFunction
from .nn_ensemble_value_function import NNEnsembleQFunction


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
        super().__init__(
            q_function.dim_state, num_states=q_function.num_states, tau=q_function.tau
        )
        self.q_function = q_function
        self.policy = policy
        self.num_samples = num_samples
        self.dist_params = dict() if dist_params is None else dist_params

    def forward(self, state):
        """Get value of the value-function at a given state."""
        pi = tensor_to_distribution(self.policy(state), **self.dist_params)
        if isinstance(self.q_function, NNEnsembleQFunction):
            out_dim = self.q_function.num_heads
        else:
            out_dim = None

        final_v = integrate(
            lambda a: self.q_function(state, a),
            pi,
            out_dim=out_dim,
            num_samples=self.num_samples,
        )
        return final_v
