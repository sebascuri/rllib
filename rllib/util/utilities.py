import numpy as np


__all__ = ['stack_list_of_tuples', 'update_parameters']


def stack_list_of_tuples(iter_, dtype=None, backend=np):
    """Convert a list of observation tuples to a list of numpy arrays.

    Parameters
    ----------
    iter_: list
        Each entry represents one row in the resulting vectors.
    dtype: type
    backend: Module
        A library that implements `backend.stack`. E.g., numpy or torch.

    Returns
    -------
    *arrays
        One stacked array for each entry in the tuple.
    """
    stacked_generator = (backend.stack(x) for x in zip(*iter_))
    if dtype is not None:
        stacked_generator = (x.astype(dtype) for x in stacked_generator)

    entry_class = iter_[0].__class__
    if entry_class in (tuple, list):
        return entry_class(stacked_generator)
    else:
        return entry_class(*stacked_generator)


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
