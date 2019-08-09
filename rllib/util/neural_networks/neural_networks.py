"""Implementation of different Neural Networks with pytorch."""

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.distributions import MultivariateNormal, Categorical
from .utilities import inverse_softplus

__all__ = ['DeterministicNN', 'ProbabilisticNN', 'HeteroGaussianNN', 'HomoGaussianNN',
           'CategoricalNN', 'EnsembleNN', 'FelixNet']


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
        covariance = torch.diag_embed(covariance)

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
        covariance = functional.softplus(self._covariance)
        covariance = torch.diag_embed(covariance)

        return MultivariateNormal(mean, covariance * self.temperature)


class CategoricalNN(ProbabilisticNN):
    """A Module that parametrizes a Categorical distribution."""

    def __init__(self, in_dim, out_dim, layers: list = None, temperature=1.0):
        super().__init__(in_dim, out_dim, layers=layers, temperature=temperature)

    def forward(self, x):
        output = self._head(self._layers(x))
        return Categorical(logits=output / self.temperature)


class EnsembleNN(ProbabilisticNN):
    """Implementation of an Ensemble of Neural Networks.

    The Ensemble shares the inner layers and then has `num_heads' different heads.
    Using these heads, it returns a Multivariate distribution parametrized by the
    mean and variance of the heads' outputs.

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
    num_heads: int
        number of heads of ensemble

    """

    def __init__(self, in_dim, out_dim, layers=None, temperature=1.0, num_heads=5):
        self.num_heads = num_heads
        super().__init__(in_dim, out_dim * num_heads, layers, temperature)

    def forward(self, x):
        x = self._layers(x)
        out = self._head(x)

        out = torch.reshape(out, out.shape[:-1] + (-1, self.num_heads))

        mean = torch.mean(out, dim=-1)
        covariance = torch.diag_embed(torch.var(out, dim=-1))

        return MultivariateNormal(mean, covariance * self.temperature)


class FelixNet(nn.Module):
    """A Module that implements FelixNet."""

    def __init__(self, in_dim, out_dim, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.layers = [64, 64]

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
        covariance = torch.diag_embed(covariance)

        return MultivariateNormal(torch.tanh(mean), covariance * self.temperature)
