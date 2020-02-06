"""TD-L1 Learning algorithm."""

from .abstract_td import AbstractTDLearning
import torch
from torch.distributions import Bernoulli


class TDL1(AbstractTDLearning):
    """TD-L1 Learning algorithm."""

    def _update(self, td_error, phi, next_phi, weight):
        weight_minus = torch.ones_like(weight) / weight
        s = Bernoulli(torch.tensor(weight / (weight_minus + weight))).sample().float()
        self.theta += self.lr_theta * s @ (phi - self.gamma * next_phi)
