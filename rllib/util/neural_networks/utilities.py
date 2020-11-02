"""Utilities for neural networks."""
import copy
import os

import numpy as np
import torch
import torch.jit
import torch.nn as nn


def deep_copy_module(module):
    """Deep copy a module."""
    if isinstance(module, torch.jit.ScriptModule):
        module.save(module.original_name)
        out = torch.jit.load(module.original_name)
        os.system(f"rm {module.original_name}")
        return out
    return copy.deepcopy(module)


class Swish(nn.Module):
    """Swish activation function.

    The swish activation function is given by:
        f(x) = x * sigmoid(x)

    References
    ----------
    Ramachandran, P., Zoph, B., & Le, Q. V. (2017).
    Swish: a self-gated activation function. arXiv.
    """

    def forward(self, x):
        """Apply forward computation of the module."""
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    """Mish activation function.

    The swish activation function is given by:
        f(x) = x * tanh(softplus(x))

    References
    ----------
    Misra, D. (2019).
    Mish: A self regularized non-monotonic neural activation function. arXiv.
    """

    def forward(self, x):
        """Apply forward computation of module."""
        return x * (torch.tanh(torch.nn.functional.softplus(x)))


class View(nn.Module):
    """View Layer.

    The  view layer flattens a tensor.
    It is useful for changing from convolutions to FC layers.
    """

    def forward(self, x):
        """Apply forward computation of module."""
        return x.view(x.size(0), -1)


def parse_nonlinearity(non_linearity):
    """Parse non-linearity."""
    if hasattr(nn, non_linearity):
        return getattr(nn, non_linearity)
    elif hasattr(nn, non_linearity.capitalize()):
        return getattr(nn, non_linearity.capitalize())
    elif hasattr(nn, non_linearity.upper()):
        return getattr(nn, non_linearity.upper())
    elif non_linearity.lower() == "swish":
        return Swish
    elif non_linearity.lower() == "mish":
        return Mish
    else:
        raise NotImplementedError(f"non-linearity {non_linearity} not implemented")


def parse_layers(layers, in_dim, non_linearity):
    """Parse layers of nn."""
    nonlinearity = parse_nonlinearity(non_linearity)
    layers_ = list()
    if len(in_dim) > 1:  # Convolutional Layers.
        w = min(in_dim[1], in_dim[2])
        k0 = max(2 * int(np.floor(np.sqrt(w) / 2)), 2)
        layers_.append(
            nn.Conv2d(
                in_channels=in_dim[0],
                out_channels=32,
                kernel_size=k0,
                stride=max(k0 // 2, 1),
            )
        )
        layers_.append(nn.BatchNorm2d(32))
        layers_.append(nn.ReLU())

        layers_.append(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=max(k0 // 2, 2),
                stride=max(k0 // 4, 1),
            )
        )
        layers_.append(nn.BatchNorm2d(64))
        layers_.append(nn.ReLU())

        layers_.append(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        )
        layers_.append(nn.BatchNorm2d(64))
        layers_.append(nn.ReLU())
        layers_.append(View())

        x = torch.randn(in_dim).unsqueeze(0)
        for module in layers_:
            x = module(x)
        in_dim = x.shape[-1]
    else:
        in_dim = in_dim[0]

    for layer in layers:
        layers_.append(nn.Linear(in_dim, layer))
        layers_.append(nonlinearity())
        in_dim = layer

    return nn.Sequential(*layers_), in_dim


def update_parameters(target_module, new_module, tau=0.0):
    """Update the parameters of target_params by those of new_params (softly).

    The parameters of target_nn are replaced by:
        target_params <- (1-tau) * (target_params) + tau * (new_params)

    Parameters
    ----------
    target_module: nn.Module
    new_module: nn.Module
    tau: float, optional

    Returns
    -------
    None.
    """
    with torch.no_grad():
        target_state_dict = target_module.state_dict()
        new_state_dict = new_module.state_dict()

        for name in target_state_dict.keys():
            if target_state_dict[name] is new_state_dict[name]:
                continue
            else:
                if target_state_dict[name].data.ndim == 0:
                    target_state_dict[name].data = new_state_dict[name].data
                else:
                    target_state_dict[name].data[:] = (
                        tau * target_state_dict[name].data
                        + (1 - tau) * new_state_dict[name].data
                    )

        # It is not necessary to load the dict again as it modifies the pointer.
        # target_module.load_state_dict(target_state_dict)


def count_vars(module):
    """Count the number of variables in a module."""
    return sum([np.prod(p.shape) for p in module.parameters()])


def zero_bias(module):
    """Zero all bias terms in the parameters.

    Parameters
    ----------
    module: module to zero the biases to.

    """
    for name, param in module.named_parameters():
        if name.endswith("bias"):
            nn.init.zeros_(param)


def init_head_bias(module, offset=0.0, delta=0.1):
    """Initialize the bias sampling u.a.r. from [offset - delta, offset + delta].

    This is useful for optimistic initialization of value functions.

    Parameters
    ----------
    module: nn.Module.
        Module to initialize head weights.
    offset: float.
        Mean of the bias.
    delta: float.
        Amplitude of bias.
    """
    for name, param in module.named_parameters():
        if name.endswith("head.bias"):
            torch.nn.init.uniform_(param, offset - delta, offset + delta)


def init_head_weight(module, mean_weight=0.1, scale_weight=0.01):
    """Initialize the head of a NN.

    The mean weights are sampled u.a.r from [-mean_weight, mean_weight].
    The scale weights are sampled u.a.r from [-scale_weight, scale_weight].

    Parameters
    ----------
    module: nn.Module.
        Module to initialize head weights.
    mean_weight: float.
        Amplitude of mean weights.
    scale_weight: float.
        Amplitude of scale weights.
    """
    for name, param in module.named_parameters():
        if name.endswith("head.weight"):
            torch.nn.init.uniform_(param, -mean_weight, mean_weight)
        elif name.endswith("scale.weight"):
            torch.nn.init.uniform_(param, -scale_weight, scale_weight)


def repeat_along_dimension(array, number, dim=0):
    """Return a view into the array with a new, repeated dimension.

    Parameters
    ----------
    array : torch.Tensor
    number : int
        The number of repeats.
    dim : int
        The dimension along which to repeat.

    Returns
    -------
    array : torch.Tensor
        A view into the input array. It has an additional dimension added at `dim` and
        the this new dimension is repeated `number` times.

    >>> import torch
    >>> array = torch.tensor([[1., 2.], [3., 4.]])

    >>> repeat_along_dimension(array, number=3, dim=0)
    tensor([[[1., 2.],
             [3., 4.]],
    <BLANKLINE>
            [[1., 2.],
             [3., 4.]],
    <BLANKLINE>
            [[1., 2.],
             [3., 4.]]])

    >>> repeat_along_dimension(array, number=3, dim=1)
    tensor([[[1., 2.],
             [1., 2.],
             [1., 2.]],
    <BLANKLINE>
            [[3., 4.],
             [3., 4.],
             [3., 4.]]])
    """
    expanded_array = array.unsqueeze(dim)
    shape = [-1] * expanded_array.dim()
    shape[dim] = number
    return expanded_array.expand(*shape)


def torch_quadratic(array, matrix):
    """Compute the quadratic form in pytorch.

    Parameters
    ----------
    array : torch.Tensor
    matrix : torch.Tensor

    Returns
    -------
    ndarray
        The quadratic form evaluated for each x.
    """
    squared_values = array * (array @ matrix)
    return torch.sum(squared_values, dim=-1, keepdim=True)


def inverse_softplus(x):
    """Inverse function to torch.functional.softplus.

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    output : torch.Tensor
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.get_default_dtype())
    return torch.log(torch.exp(x) - 1.0)


class TileCode(nn.Module):
    """Tile coding implementation.

    A tile code discretizes the environment into bins.
    Given a continuous tensor, it encodes the output as the closest tile index per dim.
    The output can be either an integer or a one-hot encode vector.

    Parameters
    ----------
    low: array_like
        Array of lower value of bins (per dimension).
    high: array_like
        Array of higher value of bins (per dimesnion)
    bins: int
        Number of bins per dimension.
    one_hot: bool, optional (default = True).
        Flag that indicates if output is one-hot encoded or not.

    Notes
    -----
    Only a same number of bins is implemented as this makes compilation easier.
    """

    def __init__(self, low, high, bins, one_hot=True):
        super().__init__()
        dim = len(low)
        assert dim == len(high)

        self.tiles = torch.stack(
            [
                torch.linspace(low_, high_, bins + 1)[1:] - (high_ - low_) / (2 * bins)
                for low_, high_ in zip(low, high)
            ],
            dim=-1,
        )

        bins = torch.tensor([bins] * dim)
        self.bins = bins + 1
        self.num_outputs = self._tuple_to_int(bins).item() + 1

        if one_hot:
            self.extra_dims = -dim + self.num_outputs
        else:
            self.extra_dims = -dim + 1
        self.one_hot = one_hot

    def _tuple_to_int(self, tuple_):
        """Convert tuple of ints into a single integer."""
        out, tuple_ = tuple_[..., 0], tuple_[..., 1:]
        i = 1
        while tuple_.shape[-1] > 0:
            out = self.bins[i] * out + tuple_[..., 0]
            tuple_ = tuple_[..., 1:]
            i += 1
        return out

    def forward(self, x):
        """Encode a vector using tile-coding."""
        if x.dim() == 0 or (x.dim() == 1 and len(self.bins) == 1):
            x = x.unsqueeze(-1)
        code = self._tuple_to_int(digitize(x, self.tiles))
        if self.one_hot:
            return one_hot_encode(code, self.num_outputs)
        return code


def digitize(tensor, bin_boundaries):
    """Implement numpy digitize using torch."""
    result = torch.zeros(tensor.shape).long()
    for boundary in bin_boundaries:
        result += (tensor > boundary).long()
    return result


class OneHotEncode(nn.Module):
    """One Hot Encoder."""

    def __init__(self, num_classes: int):
        super().__init__()
        assert num_classes > 0
        self.num_classes = num_classes

    def forward(self, x):
        """One-Hot Encode Vector."""
        return one_hot_encode(x.long(), self.num_classes)


def one_hot_encode(tensor, num_classes: int):
    """Encode a tensor using one hot encoding.

    Parameters
    ----------
    tensor: torch.Tensor of dtype torch.long

    num_classes: int
        number of classes to encode.

    Returns
    -------
    tensor: torch.Tensor of dtype torch.float
        one-hot-encoded tensor.

    Raises
    ------
    TypeError:
        if tensor not of dtype long.

    """
    tensor = tensor.long()
    if tensor.dim() == 0:
        return torch.scatter(torch.zeros(num_classes), -1, tensor, 1)
    else:
        tensor_ = tensor.reshape(-1, 1)
        out = torch.scatter(torch.zeros(tensor_.shape[0], num_classes), -1, tensor_, 1)
        return out.reshape(tensor.shape + (num_classes,))


def reverse_cumsum(tensor, dim=-1):
    """Return reversed cumsum along dimensions."""
    return torch.flip(torch.cumsum(torch.flip(tensor, (dim,)), dim), (dim,))


def reverse_cumprod(tensor, dim=-1):
    """Return reversed cumprod along dimensions."""
    return torch.flip(torch.cumprod(torch.flip(tensor, (dim,)), dim), (dim,))


def get_batch_size(tensor, base_size):
    """Get the batch size of a tensor if it is a discrete or continuous tensor.

    Parameters
    ----------
    tensor: torch.tensor.
        Tensor to identify batch size
    base_size: Tuple.
        Base size of tensor.

    Returns
    -------
    batch_size: int or None

    """
    size = tensor.shape
    if len(base_size) == 0:  # Discrete
        return tuple(size)
    else:
        return tuple(size[: -len(base_size)])


def random_tensor(discrete, dim, batch_size=None):
    """Generate a random tensor with a given dimension and batch size.

    Parameters
    ----------
    discrete: bool
        Flag that indicates if random tensor is discrete.
    dim: int
        Dimensionality of random tensor
    batch_size: int, optional
        batch size of random tensor to generate.

    Returns
    -------
    tensor: torch.tensor

    """
    if discrete:
        if batch_size:
            return torch.randint(dim, (batch_size,))
        else:
            return torch.randint(dim, ())
    else:
        if batch_size:
            return torch.randn(batch_size, dim)
        else:
            return torch.randn(dim)


def freeze_parameters(module):
    """Freeze all module parameters.

    Can be used to exclude module parameters from the graph.

    Parameters
    ----------
    module : torch.nn.Module
    """
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_parameters(module):
    """Unfreeze all module parameters.

    Can be used to include excluded module parameters in the graph.

    Parameters
    ----------
    module : torch.nn.Module
    """
    for param in module.parameters():
        param.requires_grad = True


def stop_learning(module):
    """Stop learning all module parameters.

    Can be used for early stopping parameters of the graph.

    Parameters
    ----------
    module : torch.nn.Module
    """
    for param in module.parameters():
        param.requires_grad = False
        param.grad = None


def resume_learning(module):
    """Resume learning all module parameters.

    Can be used to resume learning after early stopping parameters of the graph.

    Parameters
    ----------
    module : torch.nn.Module
    """
    for param in module.parameters():
        param.requires_grad = True
        param.grad = torch.zeros_like(param.data)


class DisableGradient(object):
    """Context manager to disable gradients temporarily.

    Gradients terms will be zero-ed, momentum terms will continue.

    Parameters
    ----------
    modules : sequence
        List of torch.nn.Module.
    """

    def __init__(self, *modules):
        self.modules = modules

    def __enter__(self):
        """Freeze the parameters."""
        for module in self.modules:
            if module is not None:
                freeze_parameters(module)

    def __exit__(self, *args):
        """Unfreeze the parameters."""
        for module in self.modules:
            if module is not None:
                unfreeze_parameters(module)


class EnableGradient(object):
    """Context manager to enables gradients temporarily.

    Gradients terms will be zero-ed, momentum terms will continue.

    Parameters
    ----------
    modules : sequence
        List of torch.nn.Module.
    """

    def __init__(self, *modules):
        self.modules = modules

    def __enter__(self):
        """Unfreeze the parameters."""
        for module in self.modules:
            if module is not None:
                unfreeze_parameters(module)

    def __exit__(self, *args):
        """Freeze the parameters."""
        for module in self.modules:
            if module is not None:
                freeze_parameters(module)


class DisableOptimizer(object):
    """Context manager to disable optimization steps temporarily.

    Gradients and momentum terms will be zero-ed.

    Parameters
    ----------
    modules : sequence
        List of torch.nn.Module.
    """

    def __init__(self, *modules):
        self.modules = modules

    def __enter__(self):
        """Freeze the parameters."""
        for module in self.modules:
            if module is not None:
                stop_learning(module)

    def __exit__(self, *args):
        """Unfreeze the parameters."""
        for module in self.modules:
            if module is not None:
                resume_learning(module)
