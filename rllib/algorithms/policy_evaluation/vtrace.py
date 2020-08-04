"""V-Trace calculation of TD-Target."""

from .retrace import ReTrace


class VTrace(ReTrace):
    r"""V-Trace target.

    The V-Trace algorithm is like Re-Trace but has a different weighting for the
    TD-Error.

    .. math:: c_s = \lambda min(1, \pi(a_s|s_s) / \mu(a_s|s_s))
    .. math:: rho_s = \lambda min(rho, \pi(a_s|s_s) / \mu(a_s|s_s))

    References
    ----------
    Espeholt, L., et al. (2018).
    IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner
    Architectures. ICML.
    """

    def __init__(self, rho_bar=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho_bar = rho_bar
        if rho_bar < 1:
            raise ValueError(r"\rho_bar must be larger or equal to 1.")

    def td(self, this_v, next_v, reward, correction):
        r"""Compute the TD error.

        The correction is c_s = \lambda \max(1, \pi/mu).
        The rho factor is \rho = \max(\rho_bar, \pi/mu) = max(\rho_bar, c_s/lambda)
        """
        td = reward + self.gamma * next_v - this_v
        if self.lambda_ == 0:  # TD-0 Algorithm
            return td
        else:
            return (correction / self.lambda_).clamp_max(self.rho_bar) * td
