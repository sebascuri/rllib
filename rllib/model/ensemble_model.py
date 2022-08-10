"""Dynamical Model parametrized with a (P or D) ensemble of Neural Networks."""

import numpy as np
import torch
import torch.jit

from rllib.util.neural_networks.neural_networks import Ensemble

from .nn_model import NNModel
from .utilities import PredictionStrategy


class EnsembleModel(NNModel):
    """Ensemble Model.

    Parameters
    ----------
    num_heads: int.
        Number of heads of the ensemble.
    prediction_strategy: str, optional (default=moment_matching).
        String that indicates how to compute the predictions of the ensemble.
    deterministic: bool, optional (default=False).
        Bool that indicates if the ensemble members are probabilistic or deterministic.

    Other Parameters
    ----------------
    See NNModel.
    """

    def __init__(
        self,
        num_heads=5,
        prediction_strategy="moment_matching",
        deterministic=False,
        *args,
        **kwargs,
    ):
        super().__init__(deterministic=deterministic, *args, **kwargs)
        self.num_heads = num_heads

        self.nn = torch.nn.ModuleList(
            [
                Ensemble(
                    num_heads=num_heads,
                    prediction_strategy=prediction_strategy,
                    deterministic=deterministic,
                    **model.kwargs,
                )
                for model in self.nn
            ]
        )

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See AbstractModel.default()."""
        return super().default(environment, *args, **kwargs)

    def sample_posterior(self) -> None:
        """Set an ensemble head."""
        self.set_head(np.random.choice(self.num_heads))

    def scale(self, state, action):
        """Get variance at a state-action pair."""
        with PredictionStrategy(self, prediction_strategy="moment_matching"):
            _, scale = self.forward(state, action[..., : self.dim_action[0]])

        return scale

    def get_decomposed_predictions(self, state, action, next_state=None):
        state_action = self.state_actions_to_input_data(state, action)
        mean_std_dim = [nn.get_decomposed_predictions(state_action) for nn in self.nn]
        return self.stack_decomposed_predictions(mean_std_dim)

    def stack_decomposed_predictions(self, mean_std_dim):
        if self.discrete_state:
            return self.stack_predictions(mean_std_dim)
        if len(mean_std_dim) == 1:  # Only 1 NN.
            mean, epistemic_tril, aleatoric_tril = mean_std_dim[0]
        else:  # There is a NN per dimension.
            mean = torch.stack(
                tuple(mean_std[0][..., 0] for mean_std in mean_std_dim), -1
            )
            epistemic_tril = torch.stack(
                tuple(mean_std[1][..., 0, 0] for mean_std in mean_std_dim), -1
            )
            epistemic_tril = torch.diag_embed(epistemic_tril)

            aleatoric_tril = torch.stack(
                tuple(mean_std[2][..., 0, 0] for mean_std in mean_std_dim), -1
            )
            aleatoric_tril = torch.diag_embed(aleatoric_tril)

        if self.deterministic:
            return mean, self.temperature * epistemic_tril, torch.zeros_like(aleatoric_tril)
        return mean, self.temperature * epistemic_tril, aleatoric_tril

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

    @property
    def is_ensemble(self):
        """Check if model is an Ensemble."""
        return True
