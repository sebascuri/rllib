from typing import Any

import numpy as np
from torch.distributions import Poisson

from .experience_replay import ExperienceReplay

class BootstrapExperienceReplay(ExperienceReplay):
    mask_distribution: Poisson
    bootstrap: bool
    def __init__(
        self,
        num_bootstraps: int = ...,
        bootstrap: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
