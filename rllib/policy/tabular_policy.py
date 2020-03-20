"""Implementation of a Tabular Policy."""

import torch
import torch.nn as nn

from rllib.util.neural_networks import one_hot_encode
from .nn_policy import NNPolicy


class TabularPolicy(NNPolicy):
    """Implement tabular policy."""

    def __init__(self, num_states, num_actions):
        super().__init__(dim_state=1, dim_action=1,
                         num_states=num_states, num_actions=num_actions,
                         biased_head=False)
        nn.init.ones_(self.nn.head.weight)

    @classmethod
    def from_other(cls, other):
        """Create new Tabular Policy from another Tabular Policy."""
        return cls(other.num_states, other.num_actions)

    @property
    def table(self):
        """Get table representation of policy."""
        return self.nn.head.weight

    def set_value(self, state, new_value):
        """Set value to value function at a given state."""
        try:
            new_value = torch.log(
                one_hot_encode(new_value, num_classes=self.num_actions) + 1e-12
            )
        except TypeError:
            pass
        self.nn.head.weight[:, state] = new_value
