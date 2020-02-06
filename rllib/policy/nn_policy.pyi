from .abstract_policy import AbstractPolicy
from typing import List, Iterator
from torch import Tensor
from torch.distributions import Distribution


class NNPolicy(AbstractPolicy):
    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = None, num_actions: int = None,
                 temperature: float = 1., layers: List[int] = None, tau: float = 1.,
                 biased_head: bool = True) -> None: ...

    def __call__(self, state: Tensor) -> Distribution: ...

    @property
    def parameters(self) -> Iterator: ...

    @parameters.setter
    def parameters(self, new_params: Iterator) -> None: ...


class FelixPolicy(NNPolicy):

    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = None, num_actions: int = None,
                 temperature: float = 1.) -> None: ...
