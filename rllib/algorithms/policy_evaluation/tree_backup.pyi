from torch import Tensor

from .abstract_td_target import AbstractTDTarget

class TreeBackupLambdaTarget(AbstractTDTarget):
    def correction(self, pi_log_prob: Tensor, mu_log_prob: Tensor) -> Tensor: ...
