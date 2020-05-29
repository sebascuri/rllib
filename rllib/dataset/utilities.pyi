from typing import Callable, List, TypeVar

import numpy as np

from .datatypes import Observation, Trajectory

T = TypeVar('T')


def stack_list_of_tuples(iter_: List[T]) -> T: ...


def average_named_tuple(named_tuple_: T) -> T: ...


def map_and_cast(fun: Callable[[T], T], iter_: List[T]) -> T: ...


def bootstrap_trajectory(trajectory: Trajectory, bootstraps: int) -> List[
    Trajectory]: ...


def batch_trajectory_to_single_trajectory(trajectory: Trajectory) -> Trajectory: ...


def concatenate_observations(observation: Observation, new_observation: Observation
                             ) -> Observation: ...


def gather_trajectories(trajectories: List[Trajectory],
                        gather_dim: int = 1) -> Observation: ...
