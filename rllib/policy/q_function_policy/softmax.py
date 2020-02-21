"""SoftMax Policy."""

from .abstract_q_function_policy import AbstractQFunctionPolicy
from torch.distributions import Categorical
import torch


class SoftMax(AbstractQFunctionPolicy):
    """Implementation of Softmax Policy.

    A soft-max policy is one that has a policy given by:
        pi(a|s) propto exp[q(s, a)]

    """

    def forward(self, *args, **kwargs):
        """See `AbstractQFunctionPolicy.forward'."""
        state = args[0]
        q_val = self.q_function(state)
        temperature = self.param()
        return Categorical(torch.softmax(q_val / temperature, dim=-1))
