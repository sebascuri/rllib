from .abstract_value_function import AbstractValueFunction, AbstractQFunction
from torch import Tensor
from typing import List, Iterator
from rllib.util.neural_networks import DeterministicNN


class NNValueFunction(AbstractValueFunction):
    dimension: int
    value_function: DeterministicNN
    def __init__(self, dim_state: int, num_states: int = None, layers: List[int] = None,
                 tau: float = 1.0, biased_head: bool=True) -> None: ...
    def __call__(self, state: Tensor, action: Tensor = None) -> Tensor: ...

    @property
    def parameters(self) -> Iterator: ...

    @parameters.setter
    def parameters(self, new_params: Iterator): ...

    def embeddings(self, state: Tensor) -> Tensor: ...

class NNQFunction(AbstractQFunction):
    q_function: DeterministicNN

    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = None, num_actions: int = None,
                 layers: List[int] = None,  tau: float = 1.0, biased_head: bool=True
                 ) -> None: ...

    def __call__(self, state: Tensor, action: Tensor = None) -> Tensor: ...

    @property
    def parameters(self) -> Iterator: ...

    @parameters.setter
    def parameters(self, new_params: Iterator): ...

    def max(self, state: Tensor) -> Tensor: ...

    def argmax(self, state: Tensor) -> Tensor: ...

    def extract_policy(self, temperature=1.0):
        """Extract the policy induced by the Q-Value function."""
        if not self.discrete_action:
            raise NotImplementedError
        else:
            policy = NNPolicy(self.dim_state, self.dim_action,
                              num_states=self.num_states,
                              num_actions=self.num_actions,
                              layers=self.q_function.layers,
                              temperature=temperature,
                              biased_head=self.q_function.head.bias is not None)
            policy.parameters = self.q_function.parameters()
            return policy
