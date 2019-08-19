"""Project Data-types."""
from collections import namedtuple
import numpy as np


class Observation(namedtuple('Observation',
                             ('state', 'action', 'reward', 'next_state', 'done'))):
    """Observation datatype."""

    def __eq__(self, other):
        """Check if two observations are equal."""
        is_equal = np.allclose(self.state, other.state)
        is_equal &= np.allclose(self.action, other.action)
        is_equal &= np.allclose(self.reward, other.reward)
        is_equal &= np.allclose(self.next_state, other.next_state)
        is_equal &= self.done == other.done

        return is_equal

    def __ne__(self, other):
        """Check if two observations are not equal."""
        return not (self == other)
