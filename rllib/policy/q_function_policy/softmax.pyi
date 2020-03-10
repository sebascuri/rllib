from .abstract_q_function_policy import AbstractQFunctionPolicy
from ..abstract_policy import AbstractPolicy
from rllib.value_function import AbstractQFunction


class SoftMax(AbstractQFunctionPolicy):
    prior: AbstractPolicy

    def __init__(self, q_function: AbstractQFunction, start: float, end: float = None,
                 decay: float = None, prior: AbstractPolicy = None): ...

    @property
    def temperature(self) -> float: ...