import numpy as np
import torch
# import torch.__spec__ as torch_mod
# from typing import Union, Iterator, TypeVar
from typing import List, TypeVar
# Module = Union[np, torch_mod]
# Dtype = Union[np.generic, np.dtype, torch.dtype]
T = TypeVar('T')

def stack_list_of_tuples(iter_: List[T], dtype: torch.dtype=None) -> T: ...
