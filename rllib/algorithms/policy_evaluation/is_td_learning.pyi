from torch import Tensor

from .abstract_td_target import AbstractTDTarget

class ImportanceSamplingOffPolicyTarget(AbstractTDTarget):
    def correction(self, pi_log_p: Tensor, behavior_log_p: Tensor) -> Tensor: ...
