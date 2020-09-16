"""Implementation Derived Models."""
import torch
import torch.nn as nn

from rllib.dataset.datatypes import Observation
from rllib.dataset.transforms import (
    ActionScaler,
    DeltaState,
    MeanFunction,
    NextStateNormalizer,
    RewardNormalizer,
    StateNormalizer,
)

from .abstract_model import AbstractModel
from .ensemble_model import EnsembleModel


class TransformedModel(AbstractModel):
    """Transformed Model computes the next state distribution."""

    def __init__(self, base_model, transformations, *args, **kwargs):
        super().__init__(
            dim_state=base_model.dim_state,
            dim_action=base_model.dim_action,
            num_states=base_model.num_states,
            num_actions=base_model.num_actions,
            model_kind=base_model.model_kind,
            *args,
            **kwargs,
        )
        self.base_model = base_model
        self.forward_transformations = nn.ModuleList(transformations)
        self.reverse_transformations = nn.ModuleList(list(reversed(transformations)))

    @classmethod
    def default(
        cls,
        environment,
        base_model=None,
        model_kind="dynamics",
        transformations=None,
        *args,
        **kwargs,
    ):
        """See AbstractModel.default()."""
        if base_model is None:
            if model_kind == "dynamics":
                base_model = EnsembleModel.default(
                    environment, deterministic=True, *args, **kwargs
                )
            elif model_kind == "rewards":
                base_model = EnsembleModel.default(
                    environment,
                    model_kind=model_kind,
                    deterministic=True,
                    *args,
                    **kwargs,
                )
            else:
                raise NotImplementedError
        if transformations is None:
            if not base_model.discrete_state:
                transformations = [
                    MeanFunction(DeltaState()),
                    StateNormalizer(),
                    ActionScaler(scale=environment.action_scale),
                    RewardNormalizer(),
                    NextStateNormalizer(),
                ]
            else:
                transformations = []
        return cls(
            base_model=base_model, transformations=transformations, *args, **kwargs
        )

    @property
    def info(self):
        """Get info of base model."""
        return self.base_model.info

    def sample_posterior(self):
        """Sample a posterior from the base model."""
        self.base_model.sample_posterior()

    def set_prediction_strategy(self, val: str) -> None:
        """Set prediction strategy."""
        self.base_model.set_prediction_strategy(val)

    def forward(self, state, action, next_state=None):
        """Predict next state distribution."""
        return self.predict(state, action[..., : self.dim_action[0]], next_state)

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

    def predict(self, state, action, next_state=None):
        """Get next_state distribution."""
        none = torch.tensor(0)
        if next_state is None:
            next_state = none
        obs = Observation(
            state, action, none, next_state, none, none, none, none, none, none
        )
        for transformation in self.forward_transformations:
            obs = transformation(obs)

        # Predict next-state
        if self.model_kind == "dynamics":
            reward, done = (none, none), none
            next_state = self.base_model(obs.state, obs.action, obs.next_state)
        elif self.model_kind == "rewards":
            reward = self.base_model(obs.state, obs.action, obs.next_state)
            next_state, done = (none, none), none
        elif self.model_kind == "termination":
            done = self.base_model(obs.state, obs.action, obs.next_state)
            next_state, reward = (none, none), (none, none)
        else:
            raise ValueError(f"{self.model_kind} not in {self.allowed_model_kind}")

        # Back-transform
        obs = Observation(
            obs.state,
            obs.action,
            reward=reward[0],
            done=done,
            next_action=none,
            log_prob_action=none,
            entropy=none,
            state_scale_tril=none,
            next_state=next_state[0],
            next_state_scale_tril=next_state[1],
            reward_scale_tril=reward[1],
        )

        for transformation in self.reverse_transformations:
            obs = transformation.inverse(obs)

        if self.model_kind == "dynamics":
            return obs.next_state, obs.next_state_scale_tril
        elif self.model_kind == "rewards":
            return obs.reward, obs.reward_scale_tril
        elif self.model_kind == "termination":
            return obs.done

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

    def set_goal(self, goal):
        """Set reward model goal."""
        self.base_model.set_goal(goal)
