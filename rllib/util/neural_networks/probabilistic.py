import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.distributions import Normal, Categorical


class GaussianNN(nn.Module):
    def __init__(self, in_dim, out_dim, layers=None):
        super(GaussianNN, self).__init__()
        self._mean = nn.Linear(in_dim, out_dim)
        self._covariance = nn.Linear(in_dim, out_dim)

    def forward(self, input_):
        mean = self._mean(input_)
        covariance = torch.exp(0.5 * self._covariance(input_))
        return Normal(mean, covariance)


class CategoricalNN(nn.Module):
    def __init__(self, in_dim, out_dim, layers=None):
        super(CategoricalNN, self).__init__()
        self._layer = nn.Linear(in_dim, out_dim)

    def forward(self, input_):
        output = self._layer(input_)
        return Categorical(functional.softmax(output, dim=-1))


