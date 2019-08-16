"""Utilities for neural networks."""

import torch
import torch.nn as nn


__all__ = ['update_parameters', 'zero_bias', 'inverse_softplus', 'one_hot_encode',
           'get_batch_size', 'random_tensor']


def update_parameters(target_params, new_params, tau=1.0):
    """Update the parameters of target_params by those of new_params (softly).

    The parameters of target_nn are replaced by:
        target_params <- (1-tau) * (target_params) + tau * (new_params)

    Parameters
    ----------
    target_params: iter
    new_params: iter
    tau: float, optional

    Returns
    -------
    None.
    """
    for target_param, new_param in zip(target_params, new_params):
        if target_param is new_param:
            continue
        else:
            new_param_ = ((1.0 - tau) * target_param.data.detach()
                          + tau * new_param.data.detach())
            target_param.data.copy_(new_param_)


def zero_bias(named_params):
    """Zero all bias terms in the parameters.

    Parameters
    ----------
    named_params: iter

    """
    for name, param in named_params:
        if name.endswith('bias'):
            nn.init.zeros_(param)


def inverse_softplus(x):
    """Inverse function to torch.functional.softplus.

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    output : torch.Tensor
    """
    return torch.log(torch.exp(x) - 1.)


def one_hot_encode(tensor, num_classes):
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
    if tensor.dtype is not torch.long:
        raise TypeError("tensor should be of type torch.long. Please call .long().")

    if tensor.dim() == 0:
        return torch.scatter(torch.zeros(num_classes), -1, tensor, 1)
    else:
        batch_size = tensor.shape[0]
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(-1)

        return torch.scatter(torch.zeros(batch_size, num_classes), -1, tensor, 1)


def get_batch_size(tensor, is_discrete=None):
    """Get the batch size of a tensor if it is a discrete or continuous tensor.

    Parameters
    ----------
    tensor: torch.tensor
        tensor to identify batch size
    is_discrete: bool
        flag that indicates if it is a discrete 1-hot vector or a continuous vector.

    Returns
    -------
    batch_size: int or None

    """
    if is_discrete is None:
        is_discrete = (tensor.dtype == torch.long)

    if tensor.dim() == 0:
        return None
    elif tensor.dim() == 1:
        if is_discrete:
            return len(tensor)
        else:
            return None
    else:
        return tensor.shape[0]


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
            return torch.randint(dim, (batch_size, ))
        else:
            return torch.randint(dim, ())
    else:
        if batch_size:
            return torch.randn(batch_size, dim)
        else:
            return torch.randn(dim)
