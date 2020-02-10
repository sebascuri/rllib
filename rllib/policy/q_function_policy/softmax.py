"""SoftMax Policy."""

from .abstract_q_function_policy import AbstractQFunctionPolicy
from torch.distributions import Categorical
import torch


class SoftMax(AbstractQFunctionPolicy):
    """Implementation of Softmax Policy.

    An epsilon greedy exploration strategy chooses the greedy strategy with probability
    1-epsilon, and a random action with probability epsilon.

    If eps_end and eps_decay are not set, then epsilon will be always eps_start.
    If not, epsilon will decay exponentially at rate eps_decay from eps_start to
    eps_end.

    """

    def __call__(self, state):
        """See `AbstractQFunctionPolicy.__call__'."""
        self.step += 1
        q_val = self.q_function(state)
        temperature = self.param(self.step)
        return Categorical(torch.softmax(q_val / temperature, dim=0))
