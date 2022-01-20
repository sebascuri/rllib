"""Dynamics implemented by a Recurrent Neural Network."""
import torch
import torch.nn
from .nn_model import NNModel
from .transformed_model import TransformedModel
from .ensemble_model import EnsembleModel
from rllib.dataset.datatypes import Observation
import torch.jit
from rllib.util.utilities import safe_cholesky


class RNNModel(NNModel):
    """Abstract RNN dynamics model, used as a basis for GRU and LSTM units.

    Parameters
    __________
    dim_hidden_state: tuple
        dimension of the hidden state
    num_layers: int
        Number of RNN layers
    layers: list
        width of layers, each layer is connected with a non-linearity.
    non_linearity: string, optional (default=Tanh).
        Neural Network non-linearity.
    input_transform: nn.Module, optional (default=None).
        Module with which to transform inputs.
    """

    def __init__(self, dim_hidden_state,
                 num_layers=1,
                 layers=(),
                 biased_head=True,
                 non_linearity="Swish",
                 initial_scale=0.5,
                 input_transform=None,
                 jit_compile=False,
                 *args,
                 **kwargs):
        self.dim_hidden_state = dim_hidden_state
        self.num_layers = num_layers
        super().__init__(layers, biased_head,
                         non_linearity, initial_scale,
                         input_transform, jit_compile,
                         *args, **kwargs)

    @classmethod
    def default(cls, dim_hidden_state, environment, *args, **kwargs):
        """Get a default model for the environment."""
        return cls(
            dim_hidden_state=dim_hidden_state,
            dim_state=kwargs.pop("dim_state", environment.dim_state),
            dim_action=kwargs.pop("dim_action", environment.dim_action),
            num_states=kwargs.pop("num_states", environment.num_states),
            num_actions=kwargs.pop("num_actions", environment.num_actions),
            dim_reward=kwargs.pop("dim_reward", environment.dim_reward),
            goal=environment.goal,
            *args,
            **kwargs,
        )

    def _get_in_dim(self):
        """Get input dim for the MLP."""
        if self.discrete_state:
            in_dim = self.dim_hidden_state
        else:
            in_dim = self.dim_hidden_state[0]

        return (in_dim,)

    def _get_rnn_in_dim(self):
        """Get input dim for the RNN."""
        if self.discrete_state:
            in_dim = self.num_states
        else:
            in_dim = self.dim_state[0]

        if self.discrete_action:
            in_dim += self.num_actions
        else:
            in_dim += self.dim_action[0]

        if self.input_transform is not None:
            in_dim = in_dim + self.input_transform.extra_dim

        return (in_dim,)


class LSTMModel(RNNModel):
    """LSTM dynamics model."""
    def __init__(self,
                 dim_hidden_state,
                 num_layers=1,
                 *args,
                 **kwargs):

        super().__init__(dim_hidden_state, num_layers, *args, **kwargs)
        self.rnn = torch.nn.LSTM(input_size=self.dim_state + self.dim_action, hidden_size=dim_hidden_state,
                                 num_layers=self.num_layers)

    def forward(self, state, action, next_state=None, prev_hidden_state=None):
        """Get Next-State distribution.

        Parameters
        ----------
        state: torch.Tensor
            current state of the system
        action: torch.Tensor
            applied acton
        next_state: torch.Tensor
            true next_state of the system
        prev_hidden_state: torch.Tensor
            previous hidden state, if None,
            the zero state is passed.

        Returns
        -------
        mean, std: torch.Tensors
            mean and std of the next_state
        final_hidden_state: torch.Tensor
            last hidden state of the system,
            only passed if previous hidden state
            is received. Required for forward
            propagation of system.
        """

        state_action = self.state_actions_to_input_data(state, action)
        state_action = state_action.permute([1, 0, 2])
        if prev_hidden_state is None:
            hidden_state, final_hidden_state, final_cell_state = self.rnn(state_action)
            hidden_state = hidden_state.permute([1, 0, 2])
            mean_std_dim = [nn(hidden_state) for nn in self.nn]
            return self.stack_predictions(mean_std_dim)
        else:
            rnn_input = torch.cat((state_action, prev_hidden_state))
            hidden_state, final_hidden_state, final_cell_state = self.rnn(rnn_input)
            hidden_state = hidden_state.permute([1, 0, 2])
            mean_std_dim = [nn(hidden_state) for nn in self.nn]
            final_hidden_state = torch.cat((final_hidden_state, final_cell_state), dim=-1)
            return self.stack_predictions(mean_std_dim), final_hidden_state

    @property
    def name(self):
        """Get Model name."""
        return f"{'Deterministic' if self.deterministic else 'Probabilistic'} LSTM"


class GRUModel(RNNModel):
    """GRU dynamics model."""
    def __init__(self,
                 dim_hidden_state,
                 num_layers=1,
                 *args,
                 **kwargs):

        super().__init__(dim_hidden_state, num_layers, *args, **kwargs)
        rnn_in_dim = self._get_rnn_in_dim()
        self.rnn = torch.nn.GRU(input_size=rnn_in_dim[0], hidden_size=dim_hidden_state[0],
                                num_layers=self.num_layers)

    def forward(self, state, action, next_state=None, prev_hidden_state=None):
        """Get Next-State distribution."""

        state_action = self.state_actions_to_input_data(state, action)
        if prev_hidden_state is None:
            state_action = state_action.permute([1, 0, 2])
            hidden_state, final_hidden_state = self.rnn(state_action)
            hidden_state = hidden_state.permute([1, 0, 2])
            mean_std_dim = [nn(hidden_state) for nn in self.nn]
            return self.stack_predictions(mean_std_dim)
        else:
            hidden_state, final_hidden_state = self.rnn(state_action, prev_hidden_state)
            hidden_state = hidden_state.permute([1, 0, 2])
            mean_std_dim = [nn(hidden_state) for nn in self.nn]
            mean, std = self.stack_predictions(mean_std_dim)
            return mean, std, final_hidden_state

    @property
    def name(self):
        """Get Model name."""
        return f"{'Deterministic' if self.deterministic else 'Probabilistic'} GRU"


class TransformedRNNModel(TransformedModel):
    """Transformed Model computes the next state distribution."""
    def __init__(self, *args, **kwargs):
        super(TransformedRNNModel, self).__init__(*args, **kwargs)

    @classmethod
    def default(
            cls,
            environment,
            dim_hidden_state,
            base_model=None,
            model_kind="dynamics",
            transformations=None,
            deterministic=True,
            *args,
            **kwargs,
    ):
        """See AbstractModel.default()."""
        if base_model is None:
            if model_kind == "dynamics":
                base_model = FullEnsembleNN.default(
                    dim_hidden_state, environment, deterministic=deterministic,
                    *args, **kwargs
                )
            else:
                raise NotImplementedError
        if transformations is None:
            transformations = []
        return cls(
            base_model=base_model, transformations=transformations, *args, **kwargs
        )

    def forward(self, state, action, next_state=None, prev_hidden_state=None):
        """Predict next state distribution."""
        return self.predict(state, action[..., : self.dim_action[0]], next_state, prev_hidden_state)

    def predict(self, state, action, next_state=None, prev_hidden_state=None):
        """Get next_state distribution."""
        none = torch.tensor(0)
        if next_state is None:
            next_state = none
        obs = Observation(
            state, action, none, next_state, none, none, none, none, none, none
        )
        for transformation in self.transformations:
            obs = transformation(obs)

        # Predict next-state
        if self.model_kind == "dynamics":
            reward, done = (none, none), none
            if prev_hidden_state is None:
                next_state = self.base_model(obs.state, obs.action, obs.next_state)
                hidden_state = None
            else:
                mean_next_state, std_next_state, hidden_state = self.base_model(obs.state, obs.action, obs.next_state,
                                                                                prev_hidden_state)
                next_state = mean_next_state, std_next_state
        else:
            raise ValueError(f"{self.model_kind} not in {self.allowed_model_kind}")

        # Back-transform
        if obs.state.shape != next_state[0].shape and isinstance(
                self.base_model, EnsembleModel
        ):
            state = obs.state.unsqueeze(-2).repeat_interleave(
                self.base_model.num_heads, -2
            )
            action = obs.action.unsqueeze(-2).repeat_interleave(
                self.base_model.num_heads, -2
            )
        else:
            state = obs.state
            action = obs.action

        obs = Observation(
            state=state,
            action=action,
            reward=reward[0],
            done=done,
            next_action=none,
            log_prob_action=none,
            entropy=none,
            state_scale_tril=none,
            next_state=next_state[0],
            next_state_scale_tril=next_state[1],
            reward_scale_tril=reward[1],
        )

        for transformation in reversed(list(self.transformations)):
            obs = transformation.inverse(obs)

        if prev_hidden_state is None:
            return obs.next_state, obs.next_state_scale_tril
        else:
            return obs.next_state, obs.next_state_scale_tril, hidden_state


class FullEnsembleNN(RNNModel):
    """Ensemble of RNNs."""
    def __init__(self,
                 dim_hidden_state,
                 num_heads=5,
                 num_layers=1,
                 gru=True,
                 prediction_strategy="moment_matching",
                 *args,
                 **kwargs):

        super().__init__(dim_hidden_state, num_layers, *args, **kwargs)
        if gru:
            self.models = torch.nn.ModuleList([
                GRUModel(dim_hidden_state, num_layers, *args, **kwargs)
                for i in range(num_heads)])

        else:
            self.models = torch.nn.ModuleList([
                LSTMModel(dim_hidden_state, num_layers, *args, **kwargs)
                for i in range(num_heads)])

        self.num_heads = num_heads
        self.head_ptr = 0
        self.head_indexes = torch.zeros(1).long()
        self.prediction_strategy = prediction_strategy

    @classmethod
    def default(cls, dim_hidden_state, environment, *args, **kwargs):
        return super().default(dim_hidden_state, environment, *args, **kwargs)

    def forward(self, state, action, next_state=None, prev_hidden_state=None):
        """Get Next-State distribution."""
        if prev_hidden_state is None:
            if self.prediction_strategy in ["moment_matching", "multi_head"]:
                mean, std = self.models[0].forward(state, action, next_state)
                out_mean = mean.unsqueeze(-1)
                out_std = torch.diagonal(std, dim1=-2, dim2=-1).unsqueeze(-1)
                for i in range(1, self.num_heads):
                    mean, std = self.models[i].forward(state, action, next_state)
                    out_mean = torch.cat((out_mean, mean.unsqueeze(-1)), dim=-1)
                    std = torch.diagonal(std, dim1=-2, dim2=-1)
                    out_std = torch.cat((out_std, std.unsqueeze(-1)), dim=-1)
                if self.prediction_strategy == "moment_matching":
                    mean = out_mean.mean(-1)
                    variance = (out_std.square() + out_mean.square()).mean(-1) - mean.square()
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
                head_idx = torch.randint(self.num_heads, size=self.num_heads).unsqueeze(-1)
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

        else:
            if self.prediction_strategy in ["moment_matching", "multi_head"]:
                mean, std, hidden_state = self.models[0].forward(state, action, next_state,
                                                                 prev_hidden_state[:, :, :, 0])
                out_mean = mean.unsqueeze(-1)
                out_std = torch.diagonal(std, dim1=-2, dim2=-1).unsqueeze(-1)
                out_hidden_state = hidden_state.unsqueeze(-1)
                for i in range(1, self.num_heads):
                    mean, std, hidden_state = self.models[i].forward(state, action, next_state,
                                                                     prev_hidden_state[:, :, :, i])
                    out_mean = torch.cat((out_mean, mean.unsqueeze(-1)), dim=-1)
                    std = torch.diagonal(std, dim1=-2, dim2=-1)
                    out_std = torch.cat((out_std, std.unsqueeze(-1)), dim=-1)
                    out_hidden_state = torch.cat((out_hidden_state, hidden_state.unsqueeze(-1)), dim=-1)
                hidden_state = out_hidden_state
                if self.prediction_strategy == "moment_matching":
                    mean = out_mean.mean(-1)
                    variance = (out_std.square() + out_mean.square()).mean(-1) - mean.square()
                    scale = safe_cholesky(torch.diag_embed(variance))
                else:
                    mean = out_mean
                    scale = out_std
                # hidden_state = out_hidden_state.mean(-1)
            elif self.prediction_strategy == "sample_head":  # TS-1
                head_ptr = torch.randint(self.num_heads, (1,))
                mean, scale, hidden_state = self.models[head_ptr].forward(state, action, next_state, prev_hidden_state)
            elif self.prediction_strategy in ["set_head", "posterior"]:  # Thompson sampling
                mean, scale, hidden_state = self.models[self.head_ptr].forward(state, action, next_state,
                                                                               prev_hidden_state)
            elif self.prediction_strategy == "set_head_idx":  # TS-INF
                mean, scale, hidden_state = self.models[self.head_idx].forward(state, action, next_state,
                                                                               prev_hidden_state)
            elif self.prediction_strategy == "sample_multiple_head":
                head_idx = torch.randint(self.num_heads, size=self.num_heads).unsqueeze(-1)
                mean, std, hidden_state = self.models[head_idx[0]].forward(state, action, next_state,
                                                                           prev_hidden_state[:, :, :, 0])
                out_mean = mean.unsqueeze(-1)
                out_std = torch.diagonal(std, dim1=-2, dim2=-1).unsqueeze(-1)
                out_hidden_state = hidden_state.unsqueeze(-1)
                for i in head_idx[1:]:
                    mean, std, hidden_state = self.models[i].forward(state, action, next_state,
                                                                     prev_hidden_state[:, :, :, i])
                    out_mean = torch.cat((out_mean, mean.unsqueeze(-1)), dim=-1)
                    std = torch.diagonal(std, dim1=-2, dim2=-1)
                    out_std = torch.cat((out_std, std.unsqueeze(-1)), dim=-1)
                    out_hidden_state = torch.cat((out_hidden_state, hidden_state.unsqueeze(-1)), dim=-1)
                hidden_state = out_hidden_state
                mean = out_mean
                scale = out_std
            else:
                raise NotImplementedError
        return mean, scale, hidden_state

    @torch.jit.export
    def set_head(self, new_head: int):
        """Set the Ensemble head.

        Parameters
        ----------
        new_head: int
            If new_head == num_heads, then forward returns the average of all heads.
            If new_head < num_heads, then forward returns the output of `new_head' head.

        Raises
        ------
        ValueError: If new_head > num_heads.
        """
        self.head_ptr = new_head
        if not (0 <= self.head_ptr < self.num_heads):
            raise ValueError("head_ptr has to be between zero and num_heads - 1.")

    @torch.jit.export
    def get_head(self) -> int:
        """Get current head."""
        return self.head_ptr

    @torch.jit.export
    def set_head_idx(self, head_indexes):
        """Set ensemble head for particles.."""
        self.head_indexes = head_indexes

    @torch.jit.export
    def get_head_idx(self):
        """Get ensemble head index."""
        return self.head_indexes

    @torch.jit.export
    def set_prediction_strategy(self, prediction: str):
        """Set ensemble prediction strategy."""
        self.prediction_strategy = prediction

    @torch.jit.export
    def get_prediction_strategy(self) -> str:
        """Get ensemble head."""
        return self.prediction_strategy

    @property
    def name(self):
        """Get Model name."""
        return f"{'Deterministic' if self.deterministic else 'Probabilistic'} RNNEnsemble."
