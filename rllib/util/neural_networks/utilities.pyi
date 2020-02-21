from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
from typing import Iterator, Tuple


def update_parameters(target_params: Iterator[Parameter],
                      new_params: Iterator[Parameter], tau: float = 1.0
                      ) -> None: ...

def zero_bias(named_params: Iterator[Tuple[str, Parameter]]) -> None: ...


def inverse_softplus(x: Tensor) -> Tensor: ...


def one_hot_encode(tensor: Tensor, num_classes: int) -> Tensor: ...


def get_batch_size(tensor: Tensor, is_discrete: bool = None) -> int: ...


def random_tensor(discrete: bool, dim: int, batch_size: int = None) -> Tensor: ...


def repeat_along_dimension(array: Tensor, number: int, dim: int = 0) -> Tensor: ...


def torch_quadratic(array: Tensor, matrix: Tensor) -> Tensor:...


def freeze_parameters(module: nn.Module) -> None: ...


def unfreeze_parameters(module: nn.Module) -> None: ...


class disable_gradient(object):
    def __init__(self, *modules) -> None: ...

    def __enter__(self) -> None: ...

    def __exit__(self, *args) -> None: ...
