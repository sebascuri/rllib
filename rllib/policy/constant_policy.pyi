from torch import Tensor

from rllib.policy import AbstractPolicy

class ConstantPolicy(AbstractPolicy):
    mean: Tensor
    std: Tensor
    logits: Tensor
