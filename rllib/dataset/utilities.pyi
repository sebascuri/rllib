from typing import List, TypeVar, Union, Callable

import numpy as np
import torch.__spec__ as torch_mod

from .datatypes import Array

T = TypeVar('T')

def get_backend(array: Array) -> Union[np, torch_mod]: ...

def stack_list_of_tuples(iter_: List[T]) -> T: ...

def map_and_cast(fun: Callable[[T], T], iter_: List[T]) -> T: ...