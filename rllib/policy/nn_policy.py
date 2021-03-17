"""Policies parametrized with Neural Networks."""

import torch
import torch.nn.functional

from rllib.util.neural_networks import (
    CategoricalNN,
    FelixNet,
    HeteroGaussianNN,
    one_hot_encode,
)

from .abstract_policy import AbstractPolicy


class NNPolicy(AbstractPolicy):
    """Implementation of a Policy implemented with a Neural Network.

    Parameters
    ----------
    layers: list, optional (default=No layers).
        width of layers, each layer is connected with a non-linearity.
    biased_head: bool, optional (default=True).
        flag that indicates if head of NN has a bias term or not.
    non_linearity: string, optional (default=Tanh).
        Neural Network non-linearity.
    input_transform: nn.Module, optional (default=None).
        Module with which to transform inputs.
    jit_compile: bool.
        Flag that indicates whether to compile or not the neural network.
    """

    def __init__(
        self,
        layers=(200, 200),
        biased_head=True,
        non_linearity="Tanh",
        squashed_output=True,
        initial_scale=0.5,
        input_transform=None,
        jit_compile=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_transform = input_transform
        in_dim = self._preprocess_input_dim()

        if self.discrete_action:
            self.nn = CategoricalNN(
                in_dim,
                (self.num_actions,),
                layers=layers,
                non_linearity=non_linearity,
                biased_head=biased_head,
            )
        else:
            self.nn = HeteroGaussianNN(
                in_dim,
                self.dim_action,
                layers=layers,
                non_linearity=non_linearity,
                biased_head=biased_head,
                squashed_output=squashed_output,
                initial_scale=initial_scale,
            )
        if jit_compile:
            self.nn = torch.jit.script(self.nn)

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See AbstractPolicy.default()."""
        return super().default(environment, *args, **kwargs)

    @classmethod
    def from_other(cls, other, copy=True):
        """Create new NN Policy from another policy."""
        new = cls(
            dim_state=other.dim_state,
            dim_action=other.dim_action,
            num_states=other.num_states,
            num_actions=other.num_actions,
            tau=other.tau,
            deterministic=other.deterministic,
            action_scale=other.action_scale,
            goal=other.goal,
            input_transform=other.input_transform,
        )
        new.nn = other.nn.__class__.from_other(other.nn, copy=copy)
        return new

    @classmethod
    def from_nn(
        cls,
        module,
        dim_state,
        dim_action,
        num_states=-1,
        num_actions=-1,
        tau=0.0,
        deterministic=False,
        action_scale=1.0,
        goal=None,
        input_transform=None,
    ):
        """Create new NN Policy from a Neural Network Implementation."""
        new = cls(
            dim_state=dim_state,
            dim_action=dim_action,
            num_states=num_states,
            num_actions=num_actions,
            tau=tau,
            deterministic=deterministic,
            action_scale=action_scale,
            goal=goal,
            input_transform=input_transform,
        )
        new.nn = module
        return new

    def _preprocess_input_dim(self):
        """Get input dimension of Neural Network."""
        if self.discrete_state:
            in_dim = (self.num_states,)
        else:
            in_dim = self.dim_state

        if hasattr(self.input_transform, "extra_dim"):
            in_dim = (in_dim[0] + getattr(self.input_transform, "extra_dim"),)

        if self.goal is not None:
            in_dim = (in_dim[0] + self.goal.shape[-1],)

        return in_dim

    @torch.jit.export
    def _preprocess_state(self, state):
        """Pre-process state before input to neural network."""
        if self.input_transform is not None:  # Apply input transform.
            state = self.input_transform(state)

        if self.discrete_state:  # One hot encode discrete states.
            state = one_hot_encode(state, num_classes=self.num_states)

        if self.goal is not None:  # concatenate goal to state.
            goal = self.goal.repeat(*state.shape[:-1], 1)
            state = torch.cat((state, goal), dim=-1)

        return state

    def forward(self, state):
        """Get distribution over actions."""
        state = self._preprocess_state(state)
        out = self.nn(state)

        if self.deterministic and not self.discrete_action:
            mean = out[0]
            dim = mean.shape[-1]
            return mean, torch.zeros(dim, dim)
        else:
            return out

    @torch.jit.export
    def embeddings(self, state):
        """Get embeddings of the value-function at a given state."""
        state = self._preprocess_state(state)

        features = self.nn.last_layer_embeddings(state)
        return features.squeeze(-1)


class FelixPolicy(NNPolicy):
    """Implementation of a NN Policy using FelixNet (designed by Felix Berkenkamp).

    Parameters
    ----------
    dim_state: Tuple
        dimension of state.
    dim_action: Tuple
        dimension of action.

    Notes
    -----
    This class is only implemented for continuous state and action spaces.

    """

    def __init__(self, jit_compile=False, *args, **kwargs):
        super().__init__(jit_compile=jit_compile, *args, **kwargs)
        self.nn = FelixNet(
            self.nn.kwargs["in_dim"],
            self.nn.kwargs["out_dim"],
            initial_scale=kwargs.get("initial_scale", 0.5),
        )
        if jit_compile:
            self.nn = torch.jit.script(self.nn)

        if self.discrete_state or self.discrete_action:
            raise ValueError("num_states and num_actions have to be set to -1.")
