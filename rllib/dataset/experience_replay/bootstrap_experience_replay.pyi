"""Implementation of an Experience Replay Buffer with a Bootstrap mask."""

from typing import List
from .experience_replay import ExperienceReplay
from torch.distributions import Poisson
import numpy as np

from rllib.dataset.transforms import AbstractTransform


class BootstrapExperienceReplay(ExperienceReplay):
    mask_distribution: Poisson

    def __init__(self, max_len: int, num_bootstraps: int = 1,
                 transformations: List[AbstractTransform] = None) -> None: ...
