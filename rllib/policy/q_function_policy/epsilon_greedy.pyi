from .abstract_q_function_policy import AbstractQFunctionPolicy

class EpsGreedy(AbstractQFunctionPolicy):
    @property
    def epsilon(self) -> float: ...
