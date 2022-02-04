from typing import Any, Optional, Tuple, Type, Union, Sequence
import torch

from .nn_model import NNModel


BASE_RNN = Union[torch.nn.LSTM, torch.nn.GRU]

class RNNModel(NNModel):
    """Abstract RNN dynamics model, used as a basis for GRU and LSTM units.

    Parameters
    ----------
    dim_hidden_state: tuple
        dimension of the hidden state
    num_layers: int
        Number of RNN layers
    layers:
    """
    dim_hidden_state: Tuple[int]
    num_layers: int
    hidden_state: Optional[torch.Tensor]
    rnn: BASE_RNN

    def __init__(
        self, dim_hidden_state: Tuple[int]=..., num_layers: int = ..., base_rnn: Type[BASE_RNN]  = ..., layers: Sequence[int] = ..., *args: Any, **kwargs: Any ,
    ) -> None: ...

    def _get_in_dim(self) -> Tuple[int] : ...

    def _get_rnn_in_dim(self) -> Tuple[int] : ...
