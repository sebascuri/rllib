"""Utilities for dataset submodule."""
from itertools import product

import numpy as np
import torch

from .datatypes import Observation


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


def average_named_tuple(named_tuple_):
    """Return an averaged named-tuple."""
    return type(named_tuple_)(*map(lambda x: x.mean().item(), named_tuple_))


def average_dataclass(dataclass_):
    """Return an averaged data-class."""
    d = []
    for val in dataclass_:
        d.append(val.mean().item())

    return type(dataclass_)(*d)


def stack_list_of_tuples(iter_, dim=None):
    """Convert a list of observation tuples to a list of numpy arrays.

    Parameters
    ----------
    iter_: list
        Each entry represents one row in the resulting vectors.
    dim: int, optional (default=0).

    Returns
    -------
    *arrays
        One stacked array for each entry in the tuple.
    """
    try:
        if dim is None:
            generator = map(torch.stack, zip(*iter_))
        else:
            generator = map(
                lambda x: torch.stack(
                    x, dim=(dim if x[0].ndim > max(dim, -dim - 1) else -1)
                ),
                zip(*iter_),
            )
        return _cast_to_iter_class(generator, iter_[0].__class__)
    except (TypeError, AttributeError):
        generator = map(np.stack, zip(*iter_))
        return _cast_to_iter_class(generator, iter_[0].__class__)


def bootstrap_trajectory(trajectory, bootstraps):
    """Bootstrap a trajectory into `bootstrap' different i.i.d. trajectories."""
    num_points = len(trajectory)
    new_trajectories = []
    for _ in range(bootstraps):
        idx = np.random.choice(num_points, num_points, replace=True)
        t = []
        for i in idx:
            t.append(trajectory[i])
        new_trajectories.append(t)

    return new_trajectories


def batch_trajectory_to_single_trajectory(trajectory):
    """Convert a batch trajectory into a single trajectory.

    A batch trajectory contains a list of batch observations, e.g., Observations with
    states that have b x h x dim_states dimensions.

    Return a Trajectory that have just 1 x dim_states.
    """
    batch_shape = trajectory[0].state.shape[:-1]
    out = []
    for batch_obs in trajectory:
        expanded_obs = Observation(
            *[k.repeat(batch_shape) if k.dim() < 1 else k for k in batch_obs]
        )
        squeezed_obs = Observation(
            *[k.reshape(-1, *k.shape[len(batch_shape) :]) for k in expanded_obs]
        )
        out += [Observation(*k) for k in zip(*squeezed_obs)]

    return out


def concatenate_observations(observation, new_observation):
    """Concatenate observations and return a new observation."""
    return Observation(
        *[
            torch.cat((a, b.unsqueeze(0)), dim=0)
            for a, b in zip(observation, new_observation)
        ]
    )


def gather_trajectories(trajectories, gather_dim=1):
    """Gather parallel trajectories.

    Parameters
    ----------
    trajectories: List[Trajectory].
    gather_dim: int, optional. (default=1).
    """
    batch_trajectories = [stack_list_of_tuples(traj) for traj in trajectories]
    trajectory = Observation(
        *map(
            lambda args: torch.cat(args, dim=gather_dim)
            if args[0].dim() > 1
            else torch.stack(args, -1),
            zip(*batch_trajectories),
        )
    )
    return trajectory


def unstack_observations(observation):
    """Unstack observations in a list."""
    in_dim = observation.reward.shape
    observations = []
    for indexes in product(*map(range, in_dim)):

        def _extract_index(tensor):
            try:
                return tensor[indexes]
            except IndexError:
                return tensor

        observations.append(Observation(*map(lambda x: _extract_index(x), observation)))

    return observations
