from typing import List, Union

from torch import Tensor
from rllib.dataset.transforms import AbstractTransform
from rllib.util.parameter_decay import ParameterDecay
from .experience_replay import ExperienceReplay


class PrioritizedExperienceReplay(ExperienceReplay):
    alpha: ParameterDecay
    beta: ParameterDecay
    epsilon: Tensor
    max_priority: float
    priorities: Tensor

    def __init__(self, max_len: int, alpha: Union[float, ParameterDecay] = 0.6,
                 beta: Union[float, ParameterDecay] = 0.4,
                 epsilon: float = 0.01, max_priority: float = 10.,
                 transformations: List[AbstractTransform] = None) -> None: ...

    def _get_priority(self, td_error: Tensor) -> Tensor: ...
