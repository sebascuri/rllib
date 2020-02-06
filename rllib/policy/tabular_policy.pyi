from . import NNPolicy
from torch import Tensor
from torch.distributions import Categorical
from typing import Union


class TabularPolicy(NNPolicy):
    def __init__(self, num_states: int, num_actions: int) -> None: ...

    def __call__(self, state: Tensor) -> Categorical: ...

    @property
    def table(self) -> Tensor: ...

    def set_value(self, state: Tensor, new_value: Union[Tensor, float]) -> None: ...
