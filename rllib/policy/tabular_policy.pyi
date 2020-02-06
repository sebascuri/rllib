from . import NNPolicy
from torch import FloatTensor, LongTensor


class TabularPolicy(NNPolicy):
    def __init__(self, num_states: int, num_actions: int) -> None: ...

    @property
    def table(self) -> FloatTensor: ...

    def set_value(self, state: LongTensor, new_value: float) -> None: ...
