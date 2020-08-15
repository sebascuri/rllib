"""Implementation Derived Models."""
import torch
import torch.nn as nn

from rllib.dataset.datatypes import Observation

from .abstract_model import AbstractModel


class TransformedModel(AbstractModel):
    """Transformed Model computes the next state distribution."""

    def __init__(self, base_model, transformations):
        super().__init__(
            dim_state=base_model.dim_state,
            dim_action=base_model.dim_action,
            num_states=base_model.num_states,
            num_actions=base_model.num_actions,
        )
        self.base_model = base_model
        self.forward_transformations = nn.ModuleList(transformations)
        self.reverse_transformations = nn.ModuleList(list(reversed(transformations)))

    def sample_posterior(self):
        """Sample a posterior from the base model."""
        self.base_model.sample_posterior()

    def set_prediction_strategy(self, val: str) -> None:
        """Set prediction strategy."""
        self.base_model.set_prediction_strategy(val)

    def forward(self, state, action):
        """Predict next state distribution."""
        return self.next_state(state, action)

    def scale(self, state, action):
        """Get epistemic scale of model."""
        none = torch.tensor(0)
        obs = Observation(state, action, none, none, none, none, none, none, none, none)
        for transformation in self.forward_transformations:
            obs = transformation(obs)

        # Predict next-state
        scale = self.base_model.scale(obs.state, obs.action)

        # Back-transform
        obs = Observation(
            obs.state,
            obs.action,
            reward=none,
            done=none,
            next_action=none,
            log_prob_action=none,
            entropy=none,
            state_scale_tril=none,
            next_state=obs.state,
            next_state_scale_tril=scale,
        )

        for transformation in self.reverse_transformations:
            obs = transformation.inverse(obs)
        return obs.next_state_scale_tril

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
        obs = Observation(
            obs.state,
            obs.action,
            reward=none,
            done=none,
            next_action=none,
            log_prob_action=none,
            entropy=none,
            state_scale_tril=none,
            next_state=next_state[0],
            next_state_scale_tril=next_state[1],
        )

        for transformation in self.reverse_transformations:
            obs = transformation.inverse(obs)
        return obs.next_state, obs.next_state_scale_tril

    @torch.jit.export
    def set_head(self, head_ptr: int):
        """Set ensemble head."""
        self.base_model.set_head(head_ptr)

    @torch.jit.export
    def get_head(self) -> int:
        """Get ensemble head."""
        return self.base_model.get_head()

    @torch.jit.export
    def set_head_idx(self, head_ptr):
        """Set ensemble head for particles."""
        self.base_model.set_head_idx(head_ptr)

    @torch.jit.export
    def get_head_idx(self):
        """Get ensemble head index."""
        return self.base_model.get_head_idx()

    @torch.jit.export
    def get_prediction_strategy(self) -> str:
        """Get ensemble head."""
        return self.base_model.get_prediction_strategy()
