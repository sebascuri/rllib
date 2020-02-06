from . import NNValueFunction, NNQFunction
from torch import LongTensor, FloatTensor


class TabularValueFunction(NNValueFunction):
    def __init__(self, num_states: int, tau: float = 1.0,
                 biased_head: bool = False) -> None: ...

    @property
    def table(self) -> FloatTensor: ...

    def set_value(self, state: LongTensor, new_value: float) -> None: ...



class TabularQFunction(NNQFunction):
    def __init__(self, num_states: int, num_actions: int, tau: float = 1.0,
                 biased_head: bool = False) -> None: ...

    def table(self) -> FloatTensor: ...

    def set_value(self, state: LongTensor, action: LongTensor, new_value: float) -> None: ...
