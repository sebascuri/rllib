"""TD-L1 Learning algorithm."""

import torch
from torch.distributions import Bernoulli

from .abstract_td import AbstractTDLearning


class TDL1(AbstractTDLearning):
    """TD-L1 Learning algorithm."""

    def _update(self, td_error, phi, next_phi, weight):
        weight_minus = torch.ones_like(weight) / weight
        s = Bernoulli(torch.tensor(weight / (weight_minus + weight))).sample()
        self.theta += self.lr_theta * s @ (phi - self.gamma * next_phi)
