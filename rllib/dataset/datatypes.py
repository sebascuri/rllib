"""Project Data-types."""
import numpy as np
import torch
from typing import NamedTuple, Union
from torch.distributions import MultivariateNormal, Categorical
from gpytorch.distributions import Delta

Array = Union[np.ndarray, torch.Tensor]
State = Union[int, float, Array]
Action = Union[int, float, Array]
Reward = Union[int, float, Array]
Done = Union[bool, Array]
Gaussian = Union[MultivariateNormal, Delta]
Distribution = Union[MultivariateNormal, Delta, Categorical]


class Observation(NamedTuple):
    """Observation datatype."""

    state: State
    action: Action
    reward: Reward = np.nan
    next_state: State = np.nan
    done: Done = True
    next_action: Action = np.nan

    @staticmethod
    def _is_equal_nan(x, y):
        if (x is np.nan) ^ (y is np.nan):  # XOR
            return False
        elif (x is not np.nan) & (y is not np.nan):
            return np.allclose(x, y)
        else:
            return True

    def __eq__(self, other):
        """Check if two observations are equal."""
        if not isinstance(other, Observation):
            return NotImplemented
        is_equal = self._is_equal_nan(self.state, other.state)
        is_equal &= self._is_equal_nan(self.action, other.action)
        is_equal &= self._is_equal_nan(self.reward, other.reward)
        is_equal &= self._is_equal_nan(self.next_state, other.next_state)
        is_equal &= self._is_equal_nan(self.next_action, other.next_action)
        is_equal &= self.done == other.done
        return is_equal

    def __ne__(self, other):
        """Check if two observations are not equal."""
        return not self.__eq__(other)
