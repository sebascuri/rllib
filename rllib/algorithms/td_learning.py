"""Implementation of LSTD algorithm."""

from abc import ABC, abstractmethod


class TDLearning(ABC):
    """Implementation of TD Learning algorithms."""

    def __init__(self):
        pass

    def train(self, batches):
        """Train using TD Learning."""
        pass

    @abstractmethod
    def _td(self, state, action, reward, next_state, done):
        raise NotImplementedError
