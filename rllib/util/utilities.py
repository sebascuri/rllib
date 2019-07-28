import numpy as np


def stack_list_of_tuples(iter_, dtype=None, backend=np):
    """Convert a list of observation tuples to a list of numpy arrays.

    Parameters
    ----------
    iter_ : list
        Each entry represents one row in the resulting vectors.
    dtype : type
    backend : Module
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