"""GTD Learning algorithm."""

from .abstract_td import AbstractTDLearning


class GTD(AbstractTDLearning):
    """GTD Learning algorithm.

    omega <- omega + lr * (td * PHI - omega)
    theta <- theta + lr * (PHI - gamma * PHI') PHI * omega

    References
    ----------
    Sutton, Richard S., Csaba Szepesvári, and Hamid Reza Maei.
    "A convergent O (n) algorithm for off-policy temporal-difference learning with
    linear function approximation."
    Advances in neural information processing systems (2008).
    """

    def _update(self, td_error, phi, next_phi, weight):
        phitw = phi @ self.omega

        self.theta += self.lr_theta * phitw @ (phi - self.gamma * next_phi)
        self.omega += self.lr_omega * (td_error @ phi - self.omega)
