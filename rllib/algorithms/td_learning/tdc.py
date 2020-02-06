"""TDC Learning algorithm."""

from .abstract_td import AbstractTDLearning


class TDC(AbstractTDLearning):
    """TDC Learning algorithm.

    omega <- omega + lr * (td - PHI * omega) * PHI
    theta <- theta + lr * (td * PHI - gamma * PHI' * PHI * omega)

    References
    ----------
    Sutton, Richard S., et al. "Fast gradient-descent methods for temporal-difference
    learning with linear function approximation." Proceedings of the 26th Annual
    International Conference on Machine Learning. ACM, 2009.
    """

    def _update(self, td_error, phi, next_phi, weight):
        phitw = phi @ self.omega

        self.theta += self.lr_theta * td_error @ phi
        self.theta -= self.lr_theta * self.gamma * phitw @ next_phi  # Correction term.
        self.omega += self.lr_omega * (td_error - phitw) @ phi
