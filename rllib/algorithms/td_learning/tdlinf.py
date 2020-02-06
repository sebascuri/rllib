"""TD-Linf Learning algorithm."""

from .abstract_td import AbstractTDLearning


class TDLinf(AbstractTDLearning):
    """TD-Linf Learning algorithm."""

    double_sample = True

    def _update(self, td_error, phi, next_phi, weight):
        self.theta += self.lr_theta * td_error @ phi
        self.theta -= self.lr_theta * self.gamma * td_error @ next_phi
