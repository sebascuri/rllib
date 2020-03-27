from typing import List, TypeVar, Callable

import numpy as np

from .datatypes import Trajectory

T = TypeVar('T')

def stack_list_of_tuples(iter_: List[T]) -> T: ...

def map_and_cast(fun: Callable[[T], T], iter_: List[T]) -> T: ...

def bootstrap_trajectory(trajectory: Trajectory, bootstraps: int) -> List[Trajectory]: ...
