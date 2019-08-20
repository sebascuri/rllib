"""Policies parametrized with tables."""


from .abstract_policy import AbstractPolicy
import torch
from torch.distributions import Categorical
from rllib.util.neural_networks import one_hot_encode


class TabularPolicy(AbstractPolicy):
    """Implement tabular policy."""

    def __init__(self, num_states, num_actions):
        super().__init__(dim_state=1, dim_action=1,
                         num_states=num_states, num_actions=num_actions)
        self._policy = torch.ones(num_states, num_actions) / num_states

    def __call__(self, state, action=None):
        return Categorical(probs=self._policy[state] / torch.sum(self._policy[state]))

    @property
    def parameters(self):
        return self._policy

    @parameters.setter
    def parameters(self, new_params):
        self._policy = new_params

    def set_value(self, state, new_value):
        try:
            new_value = one_hot_encode(new_value, num_classes=self.num_actions)
        except IndexError:
            pass
        self._policy[state] = new_value
