"""SoftMax Policy."""
from ..random_policy import RandomPolicy
from .abstract_q_function_policy import AbstractQFunctionPolicy


class SoftMax(AbstractQFunctionPolicy):
    r"""Implementation of Softmax Policy.

    A soft-max policy is one that has a policy given by:
    .. math:: \pi(a|s) \propto \rho(a|s) \exp[q(s, a) / \tau],
    where \rho(a|s) is a prior policy, usually selected at random.

    """

    def __init__(self, q_function, param, prior=None, *args, **kwargs):
        super().__init__(q_function, param)
        if prior is None:
            prior = RandomPolicy(
                q_function.dim_state,
                q_function.dim_action,
                num_states=q_function.num_states,
                num_actions=q_function.num_actions,
            )
        self.prior = prior

    @property
    def temperature(self):
        """Return temperature."""
        return self.param()

    def forward(self, state):
        """See `AbstractQFunctionPolicy.forward'."""
        q_val = self.q_function(state)
        prior = self.prior(state)
        return prior + q_val / self.temperature
