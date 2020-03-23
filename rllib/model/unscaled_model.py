"""Implementation of an Unscaled Model."""
import torch
import torch.nn as nn

from rllib.dataset.datatypes import Observation
from .abstract_model import AbstractModel


class UnscaledModel(AbstractModel):
    """Unscaled Model computes the next state distribution."""

    num_transformations: int

    def __init__(self, base_model, transformations):
        super().__init__(dim_state=base_model.dim_state,
                         dim_action=base_model.dim_action,
                         num_states=base_model.num_states,
                         num_actions=base_model.num_actions)
        self.base_model = base_model
        self.forward_transformations = nn.ModuleList(transformations)
        self.reverse_transformations = nn.ModuleList(list(reversed(transformations)))
        self.num_transformations = len(transformations)

    def forward(self, state, action):
        """Predict next state distribution."""
        # batch_size = state.shape[0:-1]
        none = torch.tensor(0)
        obs = Observation(state, action, none, none, none, none, none, none, none, none)
        for transformation in self.forward_transformations:
            obs = transformation(obs)

        # Predict next-state
        # with torch.no_grad(), gpytorch.settings.fast_pred_var():
        next_state = self.base_model(obs.state, obs.action)
        # if next_state[0].shape[-1] is not self.dim_state:
        #     mean = next_state[0].transpose(0, -1)
        #     idx = torch.arange(0, next_state[0].shape[-1])
        #     var = next_state[1][..., idx, idx].transpose(0, -1)
        #
        #     mean = mean.reshape(*batch_size, self.dim_state)
        #     var = var.reshape(*batch_size, self.dim_state)
        #     cov = torch.diag_embed(var)
        #     next_state = mean, cov

        # Back-transform
        obs = Observation(state, action, reward=none, done=none, next_action=none,
                          log_prob_action=none, entropy=none, state_scale_tril=none,
                          next_state=next_state[0],
                          next_state_scale_tril=next_state[1])

        for transformation in self.reverse_transformations:
            obs = transformation.inverse(obs)
        return obs.next_state, obs.next_state_scale_tril
