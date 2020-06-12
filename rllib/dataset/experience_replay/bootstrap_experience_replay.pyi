"""Implementation of an Experience Replay Buffer with a Bootstrap mask."""

from typing import List

import numpy as np
from torch.distributions import Poisson

from rllib.dataset.transforms import AbstractTransform

from .experience_replay import ExperienceReplay


class BootstrapExperienceReplay(ExperienceReplay):
    mask_distribution: Poisson
    bootstrap: bool

    def __init__(self, max_len: int, transformations: List[AbstractTransform] = None,
                 num_bootstraps: int = 1, bootstrap: bool = True) -> None: ...
