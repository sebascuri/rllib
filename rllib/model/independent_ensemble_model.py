"""Implementation of a model composed by an ensemble of independent models."""

import numpy as np
import torch

from rllib.util.utilities import safe_cholesky

from .abstract_model import AbstractModel
from .nn_model import NNModel
from .transformed_model import TransformedModel
from .utilities import PredictionStrategy


class IndependentEnsembleModel(AbstractModel):
    """Ensemble Model with num_heads independent neural networks.

    Parameters
    ----------
    base_model: AbstractModel
    num_heads: int.
        Number of heads of the ensemble.

    Other Parameters
    ----------------
    See NNModel.
    """

    def __init__(self, models, prediction_strategy="moment_matching", *args, **kwargs):
        super().__init__(
            dim_state=models[0].dim_state,
            dim_action=models[0].dim_action,
            dim_reward=models[0].dim_reward,
            model_kind=models[0].model_kind,
            num_states=models[0].num_states,
            num_actions=models[0].num_actions,
            *args,
            **kwargs,
        )
        self.prediction_strategy = prediction_strategy
        self.num_heads = len(models)
        self.models = models
        self.head_ptr = 0

    def forward(self, state, action, next_state=None):
        """Compute the next prediction of the ensemble."""
        if self.prediction_strategy in ["moment_matching", "multi_head"]:
            mean, std = self.models[0](state, action, next_state)
            out_mean = mean.unsqueeze(-1)
            out_std = torch.diagonal(std, dim1=-2, dim2=-1).unsqueeze(-1)
            for i in range(1, self.num_heads):
                mean, std = self.models[i].forward(state, action, next_state)
                out_mean = torch.cat((out_mean, mean.unsqueeze(-1)), dim=-1)
                std = torch.diagonal(std, dim1=-2, dim2=-1)
                out_std = torch.cat((out_std, std.unsqueeze(-1)), dim=-1)
            if self.prediction_strategy == "moment_matching":
                mean = out_mean.mean(-1)
                variance = (out_std.square() + out_mean.square()).mean(
                    -1
                ) - mean.square()
                scale = safe_cholesky(torch.diag_embed(variance))
            else:
                mean = out_mean
                scale = out_std
        elif self.prediction_strategy == "sample_head":  # TS-1
            head_ptr = torch.randint(self.num_heads, (1,))
            mean, scale = self.models[head_ptr].forward(state, action, next_state)
        elif self.prediction_strategy in ["set_head", "posterior"]:  # Thompson sampling
            mean, scale = self.models[self.head_ptr].forward(state, action, next_state)
        elif self.prediction_strategy == "set_head_idx":  # TS-INF
            mean, scale = self.models[self.head_idx].forward(state, action, next_state)
        elif self.prediction_strategy == "sample_multiple_head":
            head_idx = torch.randint(self.num_heads, size=(self.num_heads,)).unsqueeze(
                -1
            )
            mean, std = self.models[head_idx[0]].forward(state, action, next_state)
            out_mean = mean.unsqueeze(-1)
            out_std = torch.diagonal(std, dim1=-2, dim2=-1).unsqueeze(-1)
            for i in head_idx[1:]:
                mean, std = self.models[i].forward(state, action, next_state)
                out_mean = torch.cat((out_mean, mean.unsqueeze(-1)), dim=-1)
                std = torch.diagonal(std, dim1=-2, dim2=-1)
                out_std = torch.cat((out_std, std.unsqueeze(-1)), dim=-1)
            mean = out_mean
            scale = out_std
        else:
            raise NotImplementedError
        return mean, scale

    @classmethod
    def default(cls, environment, num_heads=5, *args, **kwargs):
        """See AbstractModel.default()."""
        first_member = TransformedModel.default(
            environment,
            base_model=NNModel.default(environment, *args, **kwargs),
            *args,
            **kwargs,
        )
        transformations = first_member.transformations
        models = torch.nn.ModuleList(
            [first_member]
            + [
                TransformedModel.default(
                    environment,
                    base_model=NNModel.default(environment, *args, **kwargs),
                    transformations=transformations,
                    *args,
                    **kwargs,
                )
                for _ in range(num_heads - 1)
            ]
        )
        return super().default(environment, models=models, *args, **kwargs)

    def sample_posterior(self):
        """Set an ensemble head."""
        self.set_head(np.random.choice(self.num_heads))

    def scale(self, state, action):
        """Get epistemic variance at a state-action pair."""
        with PredictionStrategy(self, prediction_strategy="moment_matching"):
            _, scale = self.forward(state, action[..., : self.dim_action[0]])

        return scale

    @torch.jit.export
    def set_head(self, head_ptr):
        """Set ensemble head."""
        self.head_ptr = head_ptr

    @torch.jit.export
    def get_head(self):
        """Get ensemble head."""
        return self.head_ptr

    @torch.jit.export
    def set_prediction_strategy(self, prediction):
        """Set ensemble prediction strategy."""
        self.prediction_strategy = prediction

    @torch.jit.export
    def get_prediction_strategy(self):
        """Get ensemble head."""
        return self.prediction_strategy

    @property
    def is_rnn(self) -> bool:
        """Check if model is an RNN."""
        return self.models[0].is_rnn
