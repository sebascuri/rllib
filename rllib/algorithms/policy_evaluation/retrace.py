"""Retrace calculation of TD-Target."""

import torch

from .abstract_td_target import AbstractTDTarget


class ReTrace(AbstractTDTarget):
    r"""Importance Sampling Off-Policy TD-Learning algorithm.

    .. math:: c_s = \lambda min(1, \pi(a_s|s_s) / \mu(a_s|s_s))

    References
    ----------
    Harutyunyan, A., Bellemare, M. G., Stepleton, T., & Munos, R. (2016).
    Q (\lambda) with Off-Policy Corrections. ALT.

    """

    def correction(self, pi_log_p, behavior_log_p):
        """Return the correction at time step t."""
        return self.lambda_ * torch.exp(pi_log_p - behavior_log_p).clamp_max(1.0)
