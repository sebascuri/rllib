"""Dynamical Model parametrized with a (P or D) ensemble of Neural Networks."""

import numpy as np
import torch
import torch.jit

from rllib.util.neural_networks import Ensemble

from .nn_model import NNModel


class EnsembleModel(NNModel):
    """Ensemble Model."""

    def __init__(
        self, num_heads, prediction_strategy="moment_matching", *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        # if deterministic

        self.nn = torch.nn.ModuleList(
            [
                Ensemble(
                    num_heads=num_heads,
                    prediction_strategy=prediction_strategy,
                    deterministic=self.deterministic,
                    **model.kwargs,
                )
                for model in self.nn
            ]
        )

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See AbstractModel.default()."""
        return super().default(
            environment,
            num_heads=kwargs.pop("num_heads", 5),
            prediction_strategy=kwargs.pop("prediction_strategy", "moment_matching"),
            *args,
            **kwargs,
        )

    def sample_posterior(self) -> None:
        """Set an ensemble head."""
        self.set_head(np.random.choice(self.num_heads))

    def scale(self, state, action):
        """Get epistemic variance at a state-action pair."""
        # Save state.
        prediction_strategy = self.get_prediction_strategy()

        # Compute epistemic uncertainty through moment matching.
        self.set_prediction_strategy("moment_matching")
        _, scale = self.forward(state, action[..., : self.dim_action[0]])

        # Set state.
        self.set_prediction_strategy(prediction_strategy)

        return scale

    @torch.jit.export
    def set_head(self, head_ptr: int):
        """Set ensemble head."""
        for nn in self.nn:
            nn.set_head(head_ptr)

    @torch.jit.export
    def get_head(self) -> int:
        """Get ensemble head."""
        return self.nn[0].get_head()

    @torch.jit.export
    def set_head_idx(self, head_ptr):
        """Set ensemble head for particles."""
        for nn in self.nn:
            nn.set_head_idx(head_ptr)

    @torch.jit.export
    def get_head_idx(self):
        """Get ensemble head index."""
        return self.nn[0].get_head_idx()

    @torch.jit.export
    def set_prediction_strategy(self, prediction: str):
        """Set ensemble prediction strategy."""
        for nn in self.nn:
            nn.set_prediction_strategy(prediction)

    @torch.jit.export
    def get_prediction_strategy(self) -> str:
        """Get ensemble head."""
        return self.nn[0].get_prediction_strategy()

    @property
    def name(self):
        """Get Model name."""
        return f"{'Deterministic' if self.deterministic else 'Probabilistic'} Ensemble"
