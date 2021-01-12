"""Value function that is computed by integrating a q-function."""

from rllib.policy import NNPolicy
from rllib.util.utilities import integrate, tensor_to_distribution

from .abstract_value_function import AbstractValueFunction
from .nn_ensemble_value_function import NNEnsembleQFunction
from .nn_value_function import NNQFunction


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

    def __init__(self, q_function, policy, num_samples=4, *args, **kwargs):
        kwargs.pop("dim_state", None)
        kwargs.pop("num_states", None)
        kwargs.pop("tau", None)
        super().__init__(
            dim_state=q_function.dim_state,
            num_states=q_function.num_states,
            tau=q_function.tau,
            *args,
            **kwargs,
        )
        self.q_function = q_function
        self.policy = policy
        self.num_samples = num_samples

    def set_policy(self, new_policy):
        """Set policy."""
        self.policy = new_policy

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See AbstractValueFunction.default."""
        q_function = NNQFunction.default(environment, *args, **kwargs)
        policy = NNPolicy.default(environment, *args, **kwargs)
        return super().default(environment, q_function=q_function, policy=policy)

    def forward(self, state):
        """Get value of the value-function at a given state."""
        pi = tensor_to_distribution(self.policy(state), **self.policy.dist_params)
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
