from abc import ABCMeta

import torch.nn as nn


class AbstractModel(nn.Module, metaclass=ABCMeta):
    dim_state: int
    dim_action: int
    dim_observation: int
    num_states: int
    num_actions: int
    num_observations: int
    discrete_state: bool
    discrete_action: bool

    def __init__(self, dim_state: int, dim_action: int, dim_observation: int = -1,
                 num_states: int = -1, num_actions: int = -1,
                 num_observations: int = -1) -> None: ...

    @property
    def name(self) -> str: ...