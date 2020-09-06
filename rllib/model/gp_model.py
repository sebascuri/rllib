"""Implementation of Gaussian Processes State-Space Models."""
import gpytorch
import torch
import torch.jit
import torch.nn

from rllib.util.gaussian_processes.gps import ExactGP, RandomFeatureGP, SparseGP
from rllib.util.gaussian_processes.utilities import add_data_to_gp, bkb, summarize_gp

from .abstract_model import AbstractModel


class ExactGPModel(AbstractModel):
    """An Exact GP State Space Model."""

    def __init__(
        self,
        state,
        action,
        target,
        mean=None,
        kernel=None,
        input_transform=None,
        max_num_points=None,
        *args,
        **kwargs,
    ):
        self._state = state
        self._action = action
        self._target = target
        self._mean = mean
        self._kernel = kernel

        dim_state = (state.shape[-1],)
        dim_action = (action.shape[-1],)
        self.max_num_points = max_num_points

        super().__init__(dim_state, dim_action, deterministic=False)
        self.input_transform = input_transform
        train_x, train_y = self.state_actions_to_train_data(state, action, target)

        likelihoods = []
        gps = []
        for train_y_i in train_y:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            gp = ExactGP(train_x, train_y_i, likelihood, mean, kernel)
            gps.append(gp)
            likelihoods.append(likelihood)

        self.likelihood = torch.nn.ModuleList(likelihoods)
        self.gp = torch.nn.ModuleList(gps)

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See AbstractModel.default()."""
        dim_state, dim_action = environment.dim_state[0], environment.dim_action[0]
        return super().default(
            environment,
            state=kwargs.pop("state", torch.zeros(dim_state)),
            action=kwargs.pop("action", torch.zeros(dim_action)),
            target=kwargs.pop("target", torch.zeros(dim_state)),
            mean=kwargs.pop("mean=", None),
            kernel=kwargs.pop("kernel", None),
            input_transform=kwargs.pop("input_transform", None),
            max_num_points=kwargs.pop("max_num_points", None),
            *args,
            **kwargs,
        )

    def forward(self, state, action, next_state=None):
        """Get next state distribution."""
        test_x = self.state_actions_to_input_data(state, action)

        if self.training:
            out = [
                likelihood(gp(gp.train_inputs[0]))
                for gp, likelihood in zip(self.gp, self.likelihood)
            ]

            mean = torch.stack(tuple(o.mean for o in out), dim=0)
            scale_tril = torch.stack(tuple(o.scale_tril for o in out), dim=0)
            return mean, scale_tril
        else:
            out = [
                likelihood(gp(test_x))
                for gp, likelihood in zip(self.gp, self.likelihood)
            ]
            mean = torch.stack(tuple(o.mean for o in out), dim=-1)

            # Sometimes, gpytorch returns negative variances due to numerical errors.
            # Hence, clamp the output variance to the noise of the likelihood.
            stddev = torch.stack(
                tuple(
                    torch.sqrt(o.variance.clamp(l.noise.item() ** 2, float("inf")))
                    for o, l in zip(out, self.likelihood)
                ),
                dim=-1,
            )
            return mean, torch.diag_embed(stddev)

    def add_data(self, state, action, target):
        """Add new data to GP-Model, independently to each GP."""
        new_x, new_y = self.state_actions_to_train_data(state, action, target)
        for i, new_y_i in enumerate(new_y):
            add_data_to_gp(self.gp[i], new_x, new_y_i)

    def summarize_gp(self, weight_function=None):
        r"""Summarize training data to GP, independently for each GP.

        Training inputs are selected by greedily maximizing
        ..math:: log det(1 + \lambda K(x, x))

        until a set of size `max_points' is built.

        References
        ----------
        Seeger, M., Williams, C., & Lawrence, N. (2003).
        Fast forward selection to speed up sparse Gaussian process regression.

        Gomes, R., & Krause, A. (2010).
        Budgeted Nonparametric Learning from Data Streams. ICML.

        """
        for gp in self.gp:
            summarize_gp(
                gp,
                self.max_num_points,
                weight_function=self._transform_weight_function(weight_function),
            )

    def _transform_weight_function(self, weight_function=None):
        """Transform weight function according to input transform of the GP."""
        if not weight_function:
            return None

        if self.input_transform is None:

            def _wf(x):
                s = x[:, : -self.dim_action]
                a = x[:, -self.dim_action :]
                return weight_function(s, a)

        else:

            def _wf(x):
                s = self.input_transform.inverse(x[:, : -self.dim_action])
                a = x[:, -self.dim_action :]
                return weight_function(s, a)

        return _wf

    # @torch.jit.export
    def state_actions_to_input_data(self, state, action):
        """Convert state-action data to the gpytorch format.

        Parameters
        ----------
        state : torch.Tensor
            [N x d_x]
        action : torch.Tensor
            [N x d_u]

        Returns
        -------
        train_x : torch.Tensor
            [N x (d_x + d_u)]
        """
        # Reshape the training inputs to fit gpytorch-batch mode
        if self.input_transform is not None:
            state = self.input_transform(state)
        train_x = torch.cat((state, action), dim=-1)
        if train_x.dim() < 2:
            train_x = train_x.unsqueeze(0)

        return train_x

    # @torch.jit.export
    def state_actions_to_train_data(self, state, action, target):
        """Convert transition data to the gpytorch format.

        Parameters
        ----------
        state : torch.Tensor
            [N x d_x]
        action : torch.Tensor
            [N x d_u]
        target : torch.Tensor
            [N x d_x]

        Returns
        -------
        train_x : torch.Tensor
            [N x (d_x + d_u)]
        train_y : torch.Tensor
            [d_x x N], contiguous array
        """
        train_x = self.state_actions_to_input_data(state, action)

        train_y = target.t().contiguous()

        return train_x, train_y


class RandomFeatureGPModel(ExactGPModel):
    """GP Model approximated by Random Fourier Features."""

    def __init__(self, num_features, approximation="RFF", *args, **kwargs):
        super().__init__(*args, **kwargs)
        gps = []
        train_x, train_y = self.state_actions_to_train_data(
            self._state, self._action, self._target
        )
        for train_y_i, likelihood in zip(train_y, self.likelihood):
            gp = RandomFeatureGP(
                train_x,
                train_y_i,
                likelihood,
                num_features=num_features,
                approximation=approximation,
                mean=self._mean,
                kernel=self._kernel,
            )
            gps.append(gp)
        self.gp = torch.nn.ModuleList(gps)
        self.approximation = approximation

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See AbstractModel.default()."""
        return super().default(
            environment,
            num_features=kwargs.pop("num_features", 128),
            approximation=kwargs.pop("approximation", "RFF"),
            *args,
            **kwargs,
        )

    @property
    def name(self):
        """Get Model name."""
        return f"{self.approximation} {super().name}"

    def sample_posterior(self):
        """Sample a set of feature vectors."""
        for gp in self.gp:
            gp.sample_features()

    def set_prediction_strategy(self, val):
        """Set GP prediction strategy."""
        if val == "posterior":
            for gp in self.gp:
                gp.full_predictive_covariance = False
        else:
            for gp in self.gp:
                gp.full_predictive_covariance = True


class SparseGPModel(ExactGPModel):
    """Sparse approximation of Exact GP models."""

    def __init__(
        self, inducing_points=None, q_bar=1, approximation="DTC", *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        gps = []
        train_x, train_y = self.state_actions_to_train_data(
            self._state, self._action, self._target
        )
        if inducing_points is None:
            inducing_points = train_x

        for train_y_i, likelihood in zip(train_y, self.likelihood):
            gp = SparseGP(
                train_x,
                train_y_i,
                likelihood,
                inducing_points,
                approximation=approximation,
                mean=self._mean,
                kernel=self._kernel,
            )
            gps.append(gp)
        self.gp = torch.nn.ModuleList(gps)
        self.approximation = approximation
        self.q_bar = q_bar

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See AbstractModel.default()."""
        return super().default(
            environment,
            inducing_points=kwargs.pop("inducing_points", None),
            num_features=kwargs.pop("q_bar", 1.0),
            approximation=kwargs.pop("approximation", "DTC"),
            *args,
            **kwargs,
        )

    @property
    def name(self):
        """Get Model name."""
        return f"{self.approximation} {super().name}"

    def add_data(self, state, action, target):
        """Add Data to GP and Re-Sparsify."""
        new_x, new_y = self.state_actions_to_train_data(state, action, target)
        for i, new_y_i in enumerate(new_y):
            arm_set = torch.cat((new_x, self.gp[i].train_inputs[0]), dim=0)
            inducing_points = bkb(self.gp[i], arm_set, q_bar=self.q_bar)
            self.gp[i].set_inducing_points(inducing_points)
            add_data_to_gp(self.gp[i], new_x, new_y_i)
