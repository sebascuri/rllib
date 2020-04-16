"""Python Script Template."""

import torch
import torch.jit

from .nn_model import NNModel
from rllib.util.neural_networks import Ensemble
from rllib.util.neural_networks.utilities import one_hot_encode


class EnsembleModel(NNModel):
    """Ensemble Model."""

    def __init__(self, dim_state, dim_action, num_heads, num_states=-1, num_actions=-1,
                 layers=None, biased_head=True, non_linearity='ReLU',
                 input_transform=None, deterministic=False):
        super().__init__(dim_state, dim_action, num_states, num_actions,
                         input_transform=input_transform)
        self.num_heads = num_heads
        # if deterministic
        self.nn = Ensemble(
            self.nn.kwargs['in_dim'], self.nn.kwargs['out_dim'], layers=layers,
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
        next_state = self.nn(state_action)
        return next_state[0], next_state[1]

    @torch.jit.export
    def select_head(self, head_ptr: int):
        """Select head of ensemble."""
        self.nn.select_head(head_ptr)

    @property
    def name(self):
        """Get Model name."""
        return f"{'Deterministic' if self.deterministic else 'Probabilistic'} Ensemble"
