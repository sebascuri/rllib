from typing import List, Union

from torch import Tensor
from rllib.dataset.transforms import AbstractTransform
from .experience_replay import ExperienceReplay
from rllib.util.parameter_decay import ParameterDecay


class EXP3ExperienceReplay(ExperienceReplay):
    eta: ParameterDecay
    gamma: ParameterDecay
    max_priority: float
    priorities: Tensor

    def __init__(self, max_len: int, eta: Union[ParameterDecay, float] = 0.1,
                 gamma: Union[ParameterDecay, float] = 0.1,
                 max_priority: float = 1.,
                 transformations: List[AbstractTransform] = None
                 ) -> None: ...

    def _update_weights(self) -> None: ...