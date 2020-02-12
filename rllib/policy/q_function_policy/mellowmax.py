"""Mellow Policy."""

from .abstract_q_function_policy import AbstractQFunctionPolicy
from torch.distributions import Categorical
import torch
import scipy.optimize
from rllib.util.utilities import mellow_max


class MellowMax(AbstractQFunctionPolicy):
    """Implementation of Mellow-max Policy.

    A mellow-max policy is one that has a policy given by:
        pi(a|s) propto exp[beta q(s, a)], where beta solves:

        sum_a exp[beta (Q(s, a) - mm)] (Q(s, a) - mm) = 0

    References
    ----------
    Asadi, Kavosh, and Michael L. Littman.
    "An alternative softmax operator for reinforcement learning."
    Proceedings of the 34th International Conference on Machine Learning. 2017.
    """

    def __call__(self, state):
        """See `AbstractQFunctionPolicy.__call__'."""
        q_value = self.q_function(state)
        omega = self.param()

        adv = q_value - mellow_max(q_value, omega)

        def f(beta_):
            """Solve for beta."""
            return (torch.exp(beta_ * adv) * adv).sum().detach().numpy()

        try:
            beta = scipy.optimize.brentq(f, a=-100, b=100)
        except ValueError:
            beta = 0

        return Categorical(torch.softmax(beta * q_value, dim=-1))
