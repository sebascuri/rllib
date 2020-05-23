from typing import List, Type, TypeVar

import torch
from torch import Tensor

from rllib.dataset.datatypes import TupleDistribution, Action
from .abstract_policy import AbstractPolicy

T = TypeVar('T', bound='NNPolicy')


class NNPolicy(AbstractPolicy):
    input_transform: torch.nn.Module
    nn: torch.nn.Module

    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = -1, num_actions: int = -1,
                 layers: List[int] = None, biased_head: bool = True,
                 non_linearity: str = 'ReLU', squashed_output: bool = True,
                 tau: float = 0., deterministic: bool = False,
                 action_scale: Action = 1.,
                 input_transform: torch.nn.Module = None) -> None: ...

    @classmethod
    def from_other(cls: Type[T], other: T) -> T: ...

    @classmethod
    def from_nn(cls: Type[T], module: torch.nn.Module, dim_state: int, dim_action: int,
                num_states: int = -1, num_actions: int = -1,
                tau: float = 0.0, deterministic: bool = False,
                action_scale: Action = 1.,
                input_transform: torch.nn.Module = None): ...

    def forward(self, *args: Tensor, **kwargs) -> TupleDistribution: ...

    def embeddings(self, state: Tensor, action: Tensor = None) -> Tensor: ...


class FelixPolicy(AbstractPolicy):

    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = -1, num_actions: int = -1,
                 tau: float = 0., deterministic: bool = False,
                 action_scale: Action = 1.,
                 input_transform: torch.nn.Module = None) -> None: ...
