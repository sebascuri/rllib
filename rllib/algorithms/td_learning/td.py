"""TD Learning algorithm."""

from .abstract_td import AbstractTDLearning


class TD(AbstractTDLearning):
    """TD Learning algorithm.

    theta <- theta + lr * td * PHI.

    References
    ----------
    Sutton, Richard S. "Learning to predict by the methods of temporal differences."
    Machine learning 3.1 (1988).
    """

    def _update(self, td_error, phi, next_phi, weight):
        self.theta += self.lr_theta * td_error @ phi
