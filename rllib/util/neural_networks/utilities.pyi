from torch import Tensor
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
