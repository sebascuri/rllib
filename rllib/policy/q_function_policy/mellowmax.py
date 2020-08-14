"""Mellow Policy."""

import scipy.optimize
import torch

from rllib.util.utilities import mellow_max

from .abstract_q_function_policy import AbstractQFunctionPolicy


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

    @property
    def omega(self):
        """Return mellow-max parameter."""
        return self.param()

    def forward(self, state):
        """See `AbstractQFunctionPolicy.forward'."""
        q_value = self.q_function(state)

        mm = mellow_max(q_value, self.omega).unsqueeze(-1)
        adv = q_value - mm
        if adv.dim() < 2:
            adv = adv.unsqueeze(0)
        beta = torch.zeros_like(mm)

        for i in range(len(mm)):

            def zero_beta(beta_, idx=i):
                """Solve for beta."""
                return (torch.exp(beta_ * adv[idx]) * adv[idx]).sum().detach().numpy()

            try:
                beta[i] = torch.tensor(scipy.optimize.brentq(zero_beta, a=-100, b=100))
            except ValueError:
                pass

        return torch.softmax(beta * q_value, dim=-1)
