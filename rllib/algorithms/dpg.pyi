from typing import Any

from .abstract_algorithm import AbstractAlgorithm

class DPG(AbstractAlgorithm):
    policy_noise: float
    noise_clip: float
    def __init__(
        self, policy_noise: float, noise_clip: float, *args: Any, **kwargs: Any,
    ) -> None: ...
