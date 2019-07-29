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


def update_parameters(source_nn, sink_nn, tau=1.0):
    """Update the parameters of source_nn by those of sink_nn (possibly softly).

    The parameters of target_nn are replaced by:
        sink_nn <- tau * (source_nn) + (1-tau) * (sink_nn)

    Parameters
    ----------
    source_nn: torch.nn.Module
    sink_nn: torch.nn.Module
    tau: float, optional

    Returns
    -------
    None.
    """
    for source_param, sink_param in zip(source_nn.parameters(), sink_nn.parameters()):
        if source_param is sink_param:
            continue
        else:
            new_param = (tau * source_param.data.detach()
                         + (1. - tau) * sink_param.data.detach())
            sink_param.data.copy_(new_param)
