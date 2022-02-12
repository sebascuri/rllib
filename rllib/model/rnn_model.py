"""Dynamics implemented by a Recurrent Neural Network."""
import torch.nn as nn

from .nn_model import NNModel


class RNNModel(NNModel):
    """Abstract RNN dynamics model, used as a basis for GRU and LSTM units.

    Parameters
    ----------
    dim_hidden_state: tuple
        dimension of the hidden state
    num_layers: int
        Number of RNN layers
    """

    def __init__(
        self,
        dim_hidden_state=(10,),
        num_layers=1,
        base_rnn=nn.GRU,
        layers=(),
        *args,
        **kwargs
    ):
        self.dim_hidden_state = dim_hidden_state
        self.num_layers = num_layers
        self.hidden_state = None
        super().__init__(layers=layers, *args, **kwargs)
        self.rnn = base_rnn(
            input_size=self._get_rnn_in_dim()[0],
            hidden_size=dim_hidden_state[0],
            num_layers=self.num_layers,
            batch_first=True,
        )

    def _get_in_dim(self):
        return self.dim_hidden_state

    def _get_rnn_in_dim(self):
        return super()._get_in_dim()

    def forward(self, state, action, next_state=None):
        """Get Next-State distribution."""
        state_action = self.state_actions_to_input_data(state, action)
        if self.training:
            hidden_state, final_hidden_state = self.rnn(state_action)
        else:
            if self.hidden_state is not None:
                hidden_state, final_hidden_state = self.rnn(
                    state_action, self.hidden_state
                )
            else:
                hidden_state, final_hidden_state = self.rnn(state_action)
            self.hidden_state = final_hidden_state

        mean_std_dim = [nn(hidden_state) for nn in self.nn]
        return self.stack_predictions(mean_std_dim)

    def reset(self):
        """Reset hidden state."""
        self.hidden_state = None

    @property
    def is_rnn(self) -> bool:
        """Check if model is an RNN."""
        return True
