"""Model implemented by a Neural Network."""
import torch

from rllib.util.neural_networks import CategoricalNN, HeteroGaussianNN, one_hot_encode

from .abstract_model import AbstractModel


class NNModel(AbstractModel):
    """Implementation of a Dynamical implemented with a Neural Network.

    Parameters
    ----------
    dim_state: Tuple
        dimension of state.
    dim_action: Tuple
        dimension of action.
    num_states: int, optional
        number of discrete states (None if state is continuous).
    num_actions: int, optional
        number of discrete actions (None if action is continuous).
    layers: list, optional
        width of layers, each layer is connected with a non-linearity.
    biased_head: bool, optional
        flag that indicates if head of NN has a bias term or not.

    """

    def __init__(
        self,
        dim_state,
        dim_action,
        num_states=-1,
        num_actions=-1,
        layers=None,
        biased_head=True,
        non_linearity="Tanh",
        initial_scale=0.5,
        input_transform=None,
        deterministic=False,
    ):
        super().__init__(
            dim_state, dim_action, num_states=num_states, num_actions=num_actions
        )
        self.input_transform = input_transform

        if self.discrete_state:
            out_dim = (self.num_states,)
        else:
            out_dim = self.dim_state

        assert len(out_dim) == 1, "No images allowed."

        if self.discrete_action:
            in_dim = (out_dim[0] + self.num_actions,)
        else:
            in_dim = (out_dim[0] + self.dim_action[0],)

        if hasattr(input_transform, "extra_dim"):
            in_dim = (in_dim[0] + getattr(input_transform, "extra_dim"),)

        if self.discrete_state:
            self.nn = CategoricalNN(
                in_dim,
                out_dim,
                layers,
                biased_head=biased_head,
                non_linearity=non_linearity,
            )
        else:
            self.nn = HeteroGaussianNN(
                in_dim,
                out_dim,
                layers,
                biased_head=biased_head,
                non_linearity=non_linearity,
                squashed_output=False,
                initial_scale=initial_scale,
            )

        self.deterministic = deterministic

    def forward(self, state, action):
        """Get Next-State distribution."""
        if self.discrete_state:
            state = one_hot_encode(state.long(), num_classes=self.num_states)
        if self.discrete_action:
            action = one_hot_encode(action.long(), num_classes=self.num_actions)

        if self.input_transform is not None:
            state = self.input_transform(state)

        state_action = torch.cat((state, action), dim=-1)
        next_state = self.nn(state_action)

        if self.deterministic:
            return next_state[0], torch.zeros_like(next_state[1])
        return next_state

    @property
    def name(self):
        """Get Model name."""
        return f"{'Deterministic' if self.deterministic else 'Probabilistic'} Ensemble"
