from numpy import ndarray
from rllib.dataset.datatypes import Observation
from typing import Tuple
from .exp3_experience_replay import EXP3Sampler

class AEXP3Sampler(EXP3Sampler):
    """Sampler for L1 Algorithm."""

    def get_batch(self, batch_size: int = None
                  ) -> Tuple[Observation, ndarray, ndarray]: ...