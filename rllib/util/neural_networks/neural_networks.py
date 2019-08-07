"""Implementation of different Neural Networks with pytorch."""

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.distributions import MultivariateNormal, Categorical, Normal
from .utilities import inverse_softplus

__all__ = ['DeterministicNN', 'ProbabilisticNN', 'HeteroGaussianNN', 'HomoGaussianNN',
           'CategoricalNN', 'FelixNet']


class DeterministicNN(nn.Module):
    """Deterministic Neural Network Implementation.

    Parameters
    ----------
    in_dim: int
        input dimension of neural network.
    out_dim: int
        output dimension of neural network.
    layers: list of int
        list of width of neural network layers, each separated with a ReLU
        non-linearity.

    """

    def __init__(self, in_dim, out_dim, layers: list = None):
        super().__init__()
        self.layers = layers or list()

        layers_ = []

        for layer in self.layers:
            layers_.append(nn.Linear(in_dim, layer))
            layers_.append(nn.ReLU())
            in_dim = layer

        self._layers = nn.Sequential(*layers_)
        self._head = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self._head(self._layers(x))


class ProbabilisticNN(DeterministicNN):
    """A Module that parametrizes a torch.distributions.Distribution distribution.

    Parameters
    ----------
    in_dim: int
        input dimension of neural network.
    out_dim: int
        output dimension of neural network.
    layers: list of int
        list of width of neural network layers, each separated with a ReLU
        non-linearity.
    temperature: float, optional
        temperature scaling of output distribution.

    """

    def __init__(self, in_dim, out_dim, layers: list = None, temperature=1.0):
        super().__init__(in_dim, out_dim, layers=layers)
        self.temperature = temperature

    def forward(self, x):
        raise NotImplementedError


class HeteroGaussianNN(ProbabilisticNN):
    """A Module that parametrizes a diagonal heteroscedastic Normal distribution."""

    def __init__(self, in_dim, out_dim, layers: list = None, temperature=1.0):
        super().__init__(in_dim, out_dim, layers=layers, temperature=temperature)
        in_dim = self._head.in_features
        self._covariance = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self._layers(x)
        mean = self._head(x)
        covariance = nn.functional.softplus(self._covariance(x))
        return MultivariateNormal(mean, covariance * self.temperature)


class HomoGaussianNN(ProbabilisticNN):
    """A Module that parametrizes a diagonal homoscedastic Normal distribution."""

    def __init__(self, in_dim, out_dim, layers: list = None, temperature=1.0):
        super().__init__(in_dim, out_dim, layers=layers, temperature=temperature)
        initial_scale = inverse_softplus(torch.ones(out_dim))
        self._covariance = nn.Parameter(initial_scale, requires_grad=True)

    def forward(self, x):
        x = self._layers(x)
        mean = self._head(x)
        covariance = functional.softplus(self._covariance(x))
        return MultivariateNormal(mean, covariance * self.temperature)


class CategoricalNN(ProbabilisticNN):
    """A Module that parametrizes a Categorical distribution."""

    def __init__(self, in_dim, out_dim, layers: list = None, temperature=1.0):
        super().__init__(in_dim, out_dim, layers=layers, temperature=temperature)

    def forward(self, x):
        output = self._head(self._layers(x))
        return Categorical(logits=output / self.temperature)


class FelixNet(nn.Module):
    """A Module that implements FelixNet."""

    def __init__(self, in_dim, out_dim, temperature=1.0):
        super().__init__()
        self.temperature = temperature

        self.linear1 = nn.Linear(in_dim, 64, bias=True)
        torch.nn.init.zeros_(self.linear1.bias)

        self.linear2 = nn.Linear(64, 64, bias=True)
        torch.nn.init.zeros_(self.linear2.bias)

        self._mean = nn.Linear(64, out_dim, bias=False)
        torch.nn.init.uniform_(self._mean.weight, -0.01, 0.01)

        self._covariance = nn.Linear(64, out_dim, bias=False)
        # torch.nn.init.uniform_(self._covariance.weight, -0.01, 0.01)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))

        mean = self._mean(x)
        covariance = functional.softplus(self._covariance(x))

        return Normal(torch.tanh(mean),
                      nn.functional.softplus(covariance))
