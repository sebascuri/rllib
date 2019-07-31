import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.distributions import MultivariateNormal, Categorical, Normal
from .utilities import inverse_softplus


class DeterministicNN(nn.Module):
    def __init__(self, in_dim, out_dim, layers=None):
        super().__init__()
        self.layers = layers

        layers_ = []
        if layers is None:
            layers = []
        for layer in layers:
            layers_.append(nn.Linear(in_dim, layer))
            layers_.append(nn.ReLU())
            in_dim = layer

        self._layers = nn.Sequential(*layers_)
        self._head = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self._head(self._layers(x))


class ProbabilisticNN(DeterministicNN):
    def __init__(self, in_dim, out_dim, layers=None, temperature=1.0):
        super().__init__(in_dim, out_dim, layers=layers)
        self.temperature = temperature

    def forward(self, x):
        raise NotImplementedError


class HeteroGaussianNN(ProbabilisticNN):
    def __init__(self, in_dim, out_dim, layers=None, temperature=1.0):
        super().__init__(in_dim, out_dim, layers=layers, temperature=temperature)
        in_dim = self._head.in_features
        self._covariance = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self._layers(x)
        mean = self._head(x)
        covariance = nn.functional.softplus(self._covariance(x))
        return MultivariateNormal(mean, covariance * self.temperature)


class HomoGaussianNN(ProbabilisticNN):
    def __init__(self, in_dim, out_dim, layers=None, temperature=1.0):
        super().__init__(in_dim, out_dim, layers=layers, temperature=temperature)
        initial_scale = inverse_softplus(torch.ones(out_dim))
        self._covariance = nn.Parameter(initial_scale, requires_grad=True)

    def forward(self, x):
        x = self._layers(x)
        mean = self._head(x)
        covariance = functional.softplus(self._covariance(x))
        return MultivariateNormal(mean, covariance * self.temperature)


class CategoricalNN(ProbabilisticNN):
    def __init__(self, in_dim, out_dim, layers=None, temperature=1.0):
        super().__init__(in_dim, out_dim, layers=layers, temperature=temperature)

    def forward(self, x):
        output = self._head(self._layers(x))
        return Categorical(logits=output / self.temperature)


class FelixNet(nn.Module):
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
