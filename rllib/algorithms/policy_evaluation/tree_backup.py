"""Tree Backup calculation of TD-Target."""

from .abstract_td_target import AbstractTDTarget


class TreeBackupLambdaTarget(AbstractTDTarget):
    r"""Tree-Backup Lambda Off-Policy TD-Learning target.

    The correction factor is given by:
    .. math:: c_s = \lambda * \pi(a_s | s_s)

    References
    ----------
    Precup, D., Sutton, R. S., & Singh, S. (2000).
    Eligibility Traces for Off-Policy Policy Evaluation. ICML.

    """

    def correction(self, pi_log_p, behavior_log_p):
        """Return the correction at time step t."""
        return self._lambda * pi_log_p
