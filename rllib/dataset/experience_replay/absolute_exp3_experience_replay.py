"""Implementation of an absolute EXP3 Experience Replay Buffer."""

import numpy as np
from torch.utils.data._utils.collate import default_collate

from .exp3_experience_replay import EXP3Sampler


class AEXP3Sampler(EXP3Sampler):
    """Sampler for L1 Algorithm."""

    def get_batch(self, batch_size=None):
        """Get a batch of data."""
        batch_size = batch_size if batch_size is not None else self.batch_size
        num = len(self)
        pprobs = self.probabilities(sign=1)
        nprobs = self.probabilities(sign=-1)
        probs = 1 / 2 * (pprobs + nprobs)
        indices = np.random.choice(num, batch_size, p=probs)

        return default_collate([self[i] for i in indices]), indices, 1 / pprobs[indices]
