from typing import Any, Optional, Union

from rllib.util.parameter_decay import ParameterDecay

from .abstract_algorithm import AbstractAlgorithm
from .abstract_mb_algorithm import AbstractMBAlgorithm
from .kl_loss import KLLoss

class BPTT(AbstractAlgorithm, AbstractMBAlgorithm):
    kl_loss: KLLoss
    def __init__(
        self,
        epsilon_mean: Union[ParameterDecay, float] = ...,
        epsilon_var: Optional[Union[ParameterDecay, float]] = ...,
        regularization: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
