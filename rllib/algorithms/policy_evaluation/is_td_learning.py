"""Importance Sampling (Propensity Score) calculation of TD-Target."""

import torch

from .abstract_td_target import AbstractTDTarget


class ImportanceSamplingOffPolicyTarget(AbstractTDTarget):
    r"""Importance Sampling Off-Policy TD-Learning algorithm.

    The correction factor is given by:

    .. math:: c_s = \pi(a_s|s_s) / \mu(a_s|s_s)

    References
    ----------
    Precup, D., Sutton, R. S., & Dasgupta, S. (2001).
    Off-policy temporal-difference learning with function approximation. ICML.

    Geist, M., & Scherrer, B. (2014).
    Off-policy Learning With Eligibility Traces: A Survey. JMLR.

    """

    def correction(self, pi_log_p, behavior_log_p):
        """Return the correction at time step t."""
        return torch.exp(pi_log_p - behavior_log_p)
