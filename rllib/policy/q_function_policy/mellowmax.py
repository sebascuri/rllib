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

    def forward(self, state):
        """See `AbstractQFunctionPolicy.forward'."""
        q_value = self.q_function(state)
        omega = self.param()

        mm = mellow_max(q_value, omega).unsqueeze(-1)
        adv = q_value - mm
        if adv.dim() < 2:
            adv = adv.unsqueeze(0)
        beta = torch.zeros_like(mm)

        for i in range(len(mm)):
            def f(beta_):
                """Solve for beta."""
                return (torch.exp(beta_ * adv[i]) * adv[i]).sum().detach().numpy()

            try:
                beta[i] = torch.tensor(scipy.optimize.brentq(f, a=-100, b=100))
            except ValueError:
                pass

        return Categorical(torch.softmax(beta * q_value, dim=-1))
