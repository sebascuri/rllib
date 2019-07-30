import torch
import torch.nn as nn


__all__ = ['update_parameters', 'zero_bias', 'inverse_softplus']


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
    """Zero all bias terms in the parameters

    Parameters
    ----------
    named_params: iter

    """
    for name, param in named_params:
        if name.endswith('bias'):
            nn.init.zeros_(param)


def inverse_softplus(input_):
    """Inverse function to torch.functional.softplus.

    Parameters
    ----------
    input_ : torch.Tensor

    Returns
    -------
    output : torch.Tensor
    """
    return torch.log(torch.exp(input_) - 1.)

