from .abstract_q_function_policy import AbstractQFunctionPolicy
from ..abstract_policy import AbstractPolicy
from rllib.value_function import AbstractQFunction
from rllib.util import ParameterDecay


class SoftMax(AbstractQFunctionPolicy):
    prior: AbstractPolicy

    def __init__(self, q_function: AbstractQFunction, param: ParameterDecay, prior: AbstractPolicy = None): ...

    @property
    def temperature(self) -> float: ...