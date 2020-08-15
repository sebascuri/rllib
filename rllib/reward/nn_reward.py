"""Model implemented by a Neural Network."""
import torch

from rllib.util.neural_networks import HeteroGaussianNN, one_hot_encode

from .abstract_reward import AbstractReward


class NNReward(AbstractReward):
    """Implementation of a Reward function implemented with a Neural Network.

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
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dim_state = dim_state
        self.dim_action = dim_action

        self.num_states = num_states if num_states is not None else -1
        self.num_actions = num_actions if num_actions is not None else -1

        self.discrete_state = self.num_states >= 0
        self.discrete_action = self.num_actions >= 0

        self.input_transform = input_transform

        out_dim = (1,)

        if self.discrete_state:
            in_dim = 2 * self.num_states
        else:
            in_dim = 2 * self.dim_state[0]
        if self.discrete_action:
            in_dim += self.num_actions
        else:
            in_dim += self.dim_action[0]

        if hasattr(input_transform, "extra_dim"):
            in_dim = in_dim + getattr(input_transform, "extra_dim")
        in_dim = (in_dim,)

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

    def forward(self, state, action, next_state):
        """Get Next-State distribution."""
        if self.discrete_state:
            state = one_hot_encode(state.long(), num_classes=self.num_states)
            next_state = one_hot_encode(next_state.long(), num_classes=self.num_states)
        if self.discrete_action:
            action = one_hot_encode(action.long(), num_classes=self.num_actions)

        if self.input_transform is not None:
            state = self.input_transform(state)

        sans = torch.cat((state, action, next_state), dim=-1)

        reward = self.nn(sans)

        if self.deterministic:
            return reward[0].squeeze(-1), torch.zeros_like(reward[1])
        return reward
