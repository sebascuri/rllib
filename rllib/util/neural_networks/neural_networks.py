"""Implementation of different Neural Networks with pytorch."""

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.jit

from .utilities import inverse_softplus, parse_layers, update_parameters


class FeedForwardNN(nn.Module):
    """Feed-Forward Neural Network Implementation.

    Parameters
    ----------
    in_dim: int
        input dimension of neural network.
    out_dim: int
        output dimension of neural network.
    layers: list of int, optional
        list of width of neural network layers, each separated with a ReLU
        non-linearity.
    biased_head: bool, optional
        flag that indicates if head of NN has a bias term or not.

    """

    def __init__(self, in_dim, out_dim, layers=None, non_linearity='ReLU',
                 biased_head=True, squashed_output=False):
        super().__init__()
        self.kwargs = {'in_dim': in_dim, 'out_dim': out_dim, 'layers': layers,
                       'non_linearity': non_linearity, 'biased_head': biased_head,
                       'squashed_output': squashed_output}

        self.hidden_layers, in_dim = parse_layers(layers, in_dim, non_linearity)
        self.embedding_dim = in_dim + 1 if biased_head else in_dim
        self.head = nn.Linear(in_dim, out_dim, bias=biased_head)
        self.squashed_output = squashed_output

    @classmethod
    def from_other(cls, other, copy=True):
        """Initialize Feedforward NN from other NN Network."""
        out = cls(**other.kwargs)
        if copy:
            update_parameters(target_params=out.parameters(),
                              new_params=other.parameters())
        return out

    def forward(self, x):
        """Execute forward computation of the Neural Network.

        Parameters
        ----------
        x: torch.Tensor.
            Tensor of size [batch_size x in_dim] where the NN is evaluated.

        Returns
        -------
        out: torch.Tensor.
            Tensor of size [batch_size x out_dim].
        """
        out = self.head(self.hidden_layers(x))
        if self.squashed_output:
            return torch.tanh(out)
        return out

    @torch.jit.export
    def last_layer_embeddings(self, x):
        """Get last layer embeddings of the Neural Network.

        Parameters
        ----------
        x: torch.Tensor.
            Tensor of size [batch_size x in_dim] where the NN is evaluated.

        Returns
        -------
        out: torch.Tensor.
            Tensor of size [batch_size x embedding_dim].
        """
        out = self.hidden_layers(x)
        if self.head.bias is not None:
            out = torch.cat((out, torch.ones(out.shape[:-1] + (1,))), dim=-1)

        return out


class DeterministicNN(FeedForwardNN):
    """Declaration of a Deterministic Neural Network."""

    pass


class HeteroGaussianNN(FeedForwardNN):
    """A Module that parametrizes a diagonal heteroscedastic Normal distribution."""

    def __init__(self, in_dim, out_dim, layers=None, non_linearity='ReLU',
                 biased_head=True, squashed_output=False):
        super().__init__(in_dim, out_dim, layers=layers, non_linearity=non_linearity,
                         biased_head=biased_head, squashed_output=squashed_output)
        in_dim = self.head.in_features
        self._scale = nn.Linear(in_dim, out_dim)
        # self._scale_tril = nn.Linear(in_dim, out_dim * out_dim, bias=biased_head)

    def forward(self, x):
        """Execute forward computation of the Neural Network.

        Parameters
        ----------
        x: torch.Tensor.
            Tensor of size [batch_size x in_dim] where the NN is evaluated.

        Returns
        -------
        mean: torch.Tensor.
            Mean of size [batch_size x out_dim].
        scale_tril: torch.Tensor.
            Cholesky factorization of covariance matrix of size.
            [batch_size x out_dim x out_dim].
        """
        x = self.hidden_layers(x)
        mean = self.head(x)
        if self.squashed_output:
            mean = torch.tanh(mean)

        # TODO: Verify if this is useful or is just the action sample that gets big.
        # If the latter is the case, consider a tanh/sigmoid constrained multivariate
        # normal distribution.
        scale = torch.diag_embed(nn.functional.softplus(self._scale(x)).clamp_max(1.))
        return mean, scale


class HomoGaussianNN(FeedForwardNN):
    """A Module that parametrizes a diagonal homoscedastic Normal distribution."""

    def __init__(self, in_dim, out_dim, layers=None, non_linearity='ReLU',
                 biased_head=True, squashed_output=False):
        super().__init__(in_dim, out_dim, layers=layers, non_linearity=non_linearity,
                         biased_head=biased_head, squashed_output=squashed_output)
        initial_scale = inverse_softplus(torch.rand(out_dim))
        self._scale = nn.Parameter(initial_scale, requires_grad=True)

    def forward(self, x):
        """Execute forward computation of the Neural Network.

        Parameters
        ----------
        x: torch.Tensor
            Tensor of size [batch_size x in_dim] where the NN is evaluated.

        Returns
        -------
        out: torch.distributions.MultivariateNormal
            Multivariate distribution with mean of size [batch_size x out_dim] and
            covariance of size [batch_size x out_dim x out_dim].
        """
        x = self.hidden_layers(x)
        mean = self.head(x)
        if self.squashed_output:
            mean = torch.tanh(mean)

        scale = torch.diag_embed(functional.softplus(self._scale))

        return mean, scale


class CategoricalNN(FeedForwardNN):
    """A Module that parametrizes a Categorical distribution."""

    def __init__(self, in_dim, out_dim, layers=None, non_linearity='ReLU',
                 biased_head=True):
        super().__init__(in_dim, out_dim, layers=layers, non_linearity=non_linearity,
                         biased_head=biased_head, squashed_output=False)
        self.kwargs.pop('squashed_output')


class Ensemble(HeteroGaussianNN):
    """Ensemble of Deterministic Neural Networks.

    The Ensemble shares the inner layers and then has `num_heads' different heads.
    Using these heads, it returns a Multivariate distribution parametrized by the
    mean and variance of the heads' outputs.

    Parameters
    ----------
    in_dim: int
        input dimension of neural network.
    out_dim: int
        output dimension of neural network.
    num_heads: int
        number of heads of ensemble
    layers: list of int
        list of width of neural network layers, each separated with a ReLU
        non-linearity.

    """

    num_heads: int
    head_ptr: int

    def __init__(self, in_dim, out_dim, num_heads, layers=None, non_linearity='ReLU',
                 biased_head=True, squashed_output=False, deterministic=True):
        super().__init__(in_dim, out_dim * num_heads, layers=layers,
                         non_linearity=non_linearity, biased_head=biased_head,
                         squashed_output=squashed_output)

        self.kwargs.update(out_dim=out_dim, num_heads=num_heads)
        self.num_heads = num_heads
        self.head_ptr = num_heads
        self.deterministic = deterministic

    @classmethod
    def from_feedforward(cls, other, num_heads):
        """Initialize from a feed-forward network."""
        return cls(**other.kwargs, num_heads=num_heads,
                   deterministic=isinstance(other, DeterministicNN))

    def forward(self, x):
        """Execute forward computation of the Neural Network.

        Parameters
        ----------
        x: torch.Tensor
            Tensor of size [batch_size x in_dim] where the NN is evaluated.

        Returns
        -------
        mean: torch.Tensor.
            Mean of size [batch_size x out_dim].
        scale_tril: torch.Tensor.
            Cholesky factorization of covariance matrix of size.
            [batch_size x out_dim x out_dim].
        """
        x = self.hidden_layers(x)
        out = self.head(x)

        out = torch.reshape(out, out.shape[:-1] + (-1, self.num_heads))

        if self.deterministic:
            scale = torch.zeros_like(out)
        else:
            scale = nn.functional.softplus(self._scale(x)).clamp(1e-3, 1.)
            scale = torch.reshape(scale, scale.shape[:-1] + (-1, self.num_heads))

        if self.head_ptr == self.num_heads and self.num_heads:
            dim, num_samples = out.shape[-2:]
            scale = torch.diag_embed(torch.mean(scale, dim=-1))

            mean = torch.mean(out, dim=-1, keepdim=True)
            sigma = (mean - out) @ (mean - out).transpose(-2, -1)
            sigma += 1e-6 * torch.eye(dim)  # Add some jitter.
            scale += torch.cholesky(sigma / (num_samples - 1))
            mean = mean.squeeze(-1)

        else:
            mean = out[..., self.head_ptr]
            scale = torch.diag_embed(scale[..., self.head_ptr])

        return mean, scale

    @torch.jit.export
    def select_head(self, new_head: int):
        """Select the Ensemble head.

        Parameters
        ----------
        new_head: int
            If new_head == num_heads, then forward returns the average of all heads.
            If new_head < num_heads, then forward returns the output of `new_head' head.

        Raises
        ------
        ValueError: If new_head > num_heads.
        """
        self.head_ptr = new_head
        if new_head > self.num_heads:
            raise ValueError(
                f"{new_head} has to be smaller or equal to {self.num_heads}.")


class MultiHeadNN(FeedForwardNN):
    """Multi-Head deterministic NN."""

    def __init__(self, in_dim, out_dim, num_heads, layers=None, non_linearity='ReLU',
                 biased_head=True, squashed_output=False):
        super().__init__(in_dim, out_dim, layers=layers,
                         non_linearity=non_linearity, biased_head=biased_head,
                         squashed_output=squashed_output)

        self.kwargs.update(num_heads=num_heads)
        self.head = nn.ModuleList(
            [nn.Linear(self.head.in_features, self.head.out_features)
             for _ in range(num_heads)])
        self.num_heads = num_heads

    @classmethod
    def from_feedforward(cls, other, num_heads):
        """Initialize from a feed-forward network."""
        return cls(**other.kwargs, num_heads=num_heads)

    def forward(self, x, i: int):
        """Execute forward computation of the Neural Network.

        Parameters
        ----------
        x: torch.Tensor.
            Tensor of size [batch_size x in_dim] where the NN is evaluated.

        Returns
        -------
        out: torch.Tensor.
            Output of size [batch_size x out_dim] of current head.
        """
        x = self.hidden_layers(x)
        return self.heads[i](x)


class FelixNet(FeedForwardNN):
    """A Module that implements FelixNet."""

    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim, layers=[64, 64], non_linearity='ReLU',
                         squashed_output=True, biased_head=False)
        self.kwargs = {'in_dim': in_dim, 'out_dim': out_dim}

        torch.nn.init.zeros_(self.hidden_layers[0].bias)
        torch.nn.init.zeros_(self.hidden_layers[2].bias)
        # torch.nn.init.uniform_(self.head.weight, -0.1, 0.1)

        self._scale_tril = nn.Linear(64, out_dim, bias=False)
        # torch.nn.init.uniform_(self._covariance.weight, -0.01, 0.01)

    def forward(self, x):
        """Execute forward computation of FelixNet.

        Parameters
        ----------
        x: torch.Tensor
            Tensor of size [batch_size x in_dim] where the NN is evaluated.

        Returns
        -------
        mean: torch.Tensor.
            Mean of size [batch_size x out_dim].
        scale_tril: torch.Tensor.
            Cholesky factorization of covariance matrix of size.
            [batch_size x out_dim x out_dim].
        """
        x = self.hidden_layers(x)

        mean = torch.tanh(self.head(x))
        scale = torch.diag_embed(functional.softplus(self._scale_tril(x)))

        return mean, scale
