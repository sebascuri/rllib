"""Project Data-types."""
import numpy as np
import torch
from typing import NamedTuple, Union
from torch.distributions import MultivariateNormal, Categorical
from gpytorch.distributions import Delta

State = Union[int, float, np.ndarray, torch.Tensor]
Action = Union[int, float, np.ndarray, torch.Tensor]
Reward = Union[int, float, np.ndarray, torch.Tensor]
Done = Union[bool, np.ndarray, torch.Tensor]
Gaussian = Union[MultivariateNormal, Delta]
Distribution = Union[MultivariateNormal, Delta, Categorical]


class Observation(NamedTuple):
    """Observation datatype."""

    state: State
    action: Action
    reward: Reward
    next_state: State
    done: Done

    def __eq__(self, other):
        """Check if two observations are equal."""
        if not isinstance(other, Observation):
            return NotImplemented
        is_equal = np.allclose(self.state, other.state)
        is_equal &= np.allclose(self.action, other.action)
        is_equal &= np.allclose(self.reward, other.reward)
        is_equal &= np.allclose(self.next_state, other.next_state)
        is_equal &= self.done == other.done
        return is_equal

    def __ne__(self, other):
        """Check if two observations are not equal."""
        return not self.__eq__(other)


class SARSAObservation(NamedTuple):
    """SARSA Observation datatype."""

    state: State
    action: Action
    reward: Reward
    next_state: State
    done: Done
    next_action: Action

    def __eq__(self, other):
        """Check if two SARSA observations are equal."""
        if not isinstance(other, SARSAObservation):
            return NotImplemented
        is_equal = np.allclose(self.state, other.state)
        is_equal &= np.allclose(self.action, other.action)
        is_equal &= np.allclose(self.reward, other.reward)
        is_equal &= np.allclose(self.next_state, other.next_state)
        is_equal &= np.allclose(self.next_action, other.next_action)
        is_equal &= self.done == other.done
        return is_equal

    def __ne__(self, other):
        """Check if two SARSA observations are not equal."""
        return not self.__eq__(other)
