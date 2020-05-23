"""Implementation Derived Models."""
import torch
import torch.nn as nn

from rllib.dataset.datatypes import Observation
from .abstract_model import AbstractModel


class TransformedModel(AbstractModel):
    """Transformed Model computes the next state distribution."""

    def __init__(self, base_model, transformations):
        super().__init__(dim_state=base_model.dim_state,
                         dim_action=base_model.dim_action,
                         num_states=base_model.num_states,
                         num_actions=base_model.num_actions)
        self.base_model = base_model
        self.forward_transformations = nn.ModuleList(transformations)
        self.reverse_transformations = nn.ModuleList(list(reversed(transformations)))

    def forward(self, state, action):
        """Predict next state distribution."""
        return self.next_state(state, action)

    # @torch.jit.export
    def next_state(self, state, action):
        """Get next_state distribution."""
        none = torch.tensor(0)
        obs = Observation(state, action, none, none, none, none, none, none, none, none)
        for transformation in self.forward_transformations:
            obs = transformation(obs)

        # Predict next-state
        next_state = self.base_model(obs.state, obs.action)

        # Back-transform
        obs = Observation(obs.state, obs.action, reward=none, done=none,
                          next_action=none, log_prob_action=none, entropy=none,
                          state_scale_tril=none,
                          next_state=next_state[0],
                          next_state_scale_tril=next_state[1])

        for transformation in self.reverse_transformations:
            obs = transformation.inverse(obs)
        return obs.next_state, obs.next_state_scale_tril


class ExpectedModel(TransformedModel):
    """Expected Model returns a Delta at the expected next state."""

    def forward(self, state, action):
        """Get Expected Next state."""
        ns, ns_scale_tril = self.next_state(state, action)

        return ns, torch.zeros_like(ns_scale_tril), ns_scale_tril


class OptimisticModel(TransformedModel):
    """Optimistic Model returns a Delta at the optimistic next state."""

    def __init__(self, base_model, transformations, beta=1.0):
        super().__init__(base_model, transformations)
        self.dim_action = self.dim_action + self.dim_state
        self.beta = beta

    def forward(self, state, action):
        """Get Optimistic Next state."""
        control_action = action[..., :-self.dim_state]
        optimism_vars = action[..., -self.dim_state:]
        optimism_vars = torch.clamp(optimism_vars, -1., 1.)

        mean, tril = self.next_state(state, control_action)
        return (mean + self.beta * (tril @ optimism_vars.unsqueeze(-1)).squeeze(-1),
                torch.zeros_like(tril), tril)
