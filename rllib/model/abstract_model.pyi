from abc import ABCMeta
import torch.nn as nn


class AbstractModel(nn.Module, metaclass=ABCMeta):
    dim_state: int
    dim_action: int
    dim_observation: int
    num_states: int
    num_actions: int
    num_observations: int

    def __init__(self, dim_state: int, dim_action: int, dim_observation: int = None,
                 num_states: int = None, num_actions: int = None,
                 num_observations: int = None) -> None: ...

    @property
    def discrete_state(self) -> bool: ...

    @property
    def discrete_action(self) -> bool: ...

    @property
    def discrete_observation(self) -> bool: ...
