"""Dynamical Model parametrized with a (P or D) ensemble of Neural Networks."""

import torch
import torch.jit

from .nn_model import NNModel
from rllib.util.neural_networks import Ensemble
from rllib.util.neural_networks.utilities import one_hot_encode


class EnsembleModel(NNModel):
    """Ensemble Model."""

    def __init__(self, dim_state, dim_action, num_heads, num_states=-1, num_actions=-1,
                 prediction_strategy='moment_matching',
                 layers=None, biased_head=True, non_linearity='ReLU',
                 input_transform=None, deterministic=False):
        super().__init__(dim_state, dim_action, num_states, num_actions,
                         input_transform=input_transform)
        self.num_heads = num_heads
        # if deterministic
        self.nn = Ensemble(
            self.nn.kwargs['in_dim'], self.nn.kwargs['out_dim'],
            prediction_strategy=prediction_strategy,
            layers=layers,
            biased_head=biased_head, non_linearity=non_linearity, squashed_output=False,
            num_heads=num_heads, deterministic=deterministic)
        self.deterministic = deterministic

    def forward(self, state, action):
        """Compute next state distribution."""
        if self.discrete_state:
            state = one_hot_encode(state.long(), num_classes=self.num_states)
        if self.discrete_action:
            action = one_hot_encode(action.long(), num_classes=self.num_actions)

        if self.input_transform is not None:
            state = self.input_transform(state)

        state_action = torch.cat((state, action), dim=-1)
        return self.nn(state_action)

    @torch.jit.export
    def set_head(self, head_ptr: int):
        """Set ensemble head."""
        self.nn.set_head(head_ptr)

    @torch.jit.export
    def get_head(self) -> int:
        """Get ensemble head."""
        return self.nn.get_head()

    @torch.jit.export
    def set_head_idx(self, head_ptr):
        """Set ensemble head for particles.."""
        self.nn.set_head_idx(head_ptr)

    @torch.jit.export
    def get_head_idx(self):
        """Get ensemble head index."""
        return self.nn.get_head_idx()

    @torch.jit.export
    def set_prediction_strategy(self, prediction: str):
        """Set ensemble prediction strategy."""
        self.nn.set_prediction_strategy(prediction)

    @torch.jit.export
    def get_prediction_strategy(self) -> str:
        """Get ensemble head."""
        return self.nn.get_prediction_strategy()

    @property
    def name(self):
        """Get Model name."""
        return f"{'Deterministic' if self.deterministic else 'Probabilistic'} Ensemble"
