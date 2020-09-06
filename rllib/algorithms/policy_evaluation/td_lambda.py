"""TD-Lambda/Q-Lambda calculation of TD-Target."""

from .abstract_td_target import AbstractTDTarget


class TDLambdaTarget(AbstractTDTarget):
    r"""TD-Lambda target calculation.

    The correction factor is given by:
    .. math:: c_s = \lambda

    References
    ----------
    Harutyunyan, A., Bellemare, M. G., Stepleton, T., & Munos, R. (2016).
    Q (\lambda) with Off-Policy Corrections. ALT.

    """

    def correction(self, pi_log_p, behavior_log_p):
        """Return the correction at time step t."""
        return self._lambda
