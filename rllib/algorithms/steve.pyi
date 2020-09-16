from torch import Tensor

from rllib.dataset.datatypes import Loss, Observation

from .dyna import Dyna

class STEVE(Dyna):
    num_models: int
    num_q = int
    def __init__(self) -> None: ...
    def model_augmented_critic_loss(self, observation: Observation) -> Loss: ...
    def get_value_target(self, observation: Observation) -> Tensor: ...
