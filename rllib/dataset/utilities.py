"""Utilities for dataset submodule."""
import torch
import numpy as np


def get_backend(array):
    """Get backend of the array."""
    if isinstance(array, torch.Tensor):
        return torch
    else:
        return np


def _cast_to_iter_class(generator, class_):
    if class_ in (tuple, list):
        return class_(generator)
    else:
        return class_(*generator)


def map_and_cast(fun, iter_):
    """Map a function on a iterable and recast the resulting generator.

    Parameters
    ----------
    fun : callable
    iter_ : iterable
    """
    generator = map(fun, zip(*iter_))
    return _cast_to_iter_class(generator, iter_[0].__class__)


def stack_list_of_tuples(iter_):
    """Convert a list of observation tuples to a list of numpy arrays.

    Parameters
    ----------
    iter_: list
        Each entry represents one row in the resulting vectors.

    Returns
    -------
    *arrays
        One stacked array for each entry in the tuple.
    """
    try:
        generator = map(torch.stack, zip(*iter_))
        return _cast_to_iter_class(generator, iter_[0].__class__)
    except TypeError:
        generator = map(np.stack, zip(*iter_))
        return _cast_to_iter_class(generator, iter_[0].__class__)
