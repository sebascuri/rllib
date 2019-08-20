"""Policies parametrized with tables."""


from . import NNPolicy
import torch.nn as nn
from rllib.util.neural_networks import one_hot_encode


class TabularPolicy(NNPolicy):
    """Implement tabular policy."""

    def __init__(self, num_states, num_actions):
        super().__init__(dim_state=1, dim_action=1,
                         num_states=num_states, num_actions=num_actions,
                         temperature=0, biased_head=False)
        nn.init.ones_(self._policy.head.weight)

    @property
    def table(self):
        return self._policy.head.weight

    def set_value(self, state, new_value):
        try:
            new_value = one_hot_encode(new_value, num_classes=self.num_actions)
        except TypeError:
            pass
        self._policy.head.weight[:, state] = new_value
