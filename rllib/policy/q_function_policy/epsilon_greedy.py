"""Epsilon Greedy Policy."""

from .abstract_q_function_policy import AbstractQFunctionPolicy
from torch.distributions import Categorical
import torch


class EpsGreedy(AbstractQFunctionPolicy):
    """Implementation of Epsilon Greedy Policy.

    An epsilon greedy exploration strategy chooses the greedy strategy with probability
    1-epsilon, and a random action with probability epsilon.

    If eps_end and eps_decay are not set, then epsilon will be always eps_start.
    If not, epsilon will decay exponentially at rate eps_decay from eps_start to
    eps_end.

    """

    def __call__(self, state):
        """See `AbstractQFunctionPolicy.__call__'."""
        # Epsilon part.
        epsilon = self.param()
        probs = epsilon / self.num_actions * torch.ones(self.num_actions)

        # Greedy part.
        a = torch.argmax(self.q_function(state))
        probs[a] += (1 - epsilon)
        return Categorical(probs)
