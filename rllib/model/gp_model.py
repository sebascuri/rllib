"""Implementation of Gaussian Processes State-Space Models."""
import gpytorch
import torch
import torch.jit
import torch.nn

from rllib.util.gaussian_processes.gps import ExactGP, SparseGP, RandomFeatureGP
from rllib.util.gaussian_processes.utilities import add_data_to_gp, summarize_gp, bkb
from .abstract_model import AbstractModel


class ExactGPModel(AbstractModel):
    """An Exact GP State Space Model."""

    def __init__(self, state, action, next_state, mean=None, kernel=None,
                 input_transform=None, max_num_points=None):
        dim_state = state.shape[-1]
        dim_action = action.shape[-1]
        self.max_num_points = max_num_points

        super().__init__(dim_state, dim_action)
        self.input_transform = input_transform
        train_x, train_y = self.state_actions_to_train_data(state, action, next_state)

        likelihoods = []
        gps = []
        for train_y_i in train_y:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            gp = ExactGP(train_x, train_y_i, likelihood, mean, kernel)
            gps.append(gp)
            likelihoods.append(likelihood)

        self.likelihood = torch.nn.ModuleList(likelihoods)
        self.gp = torch.nn.ModuleList(gps)

    def forward(self, state, action):
        """Get next state distribution."""
        test_x = self.state_actions_to_input_data(state, action)

        out = [likelihood(gp(test_x))
               for gp, likelihood in zip(self.gp, self.likelihood)]

        if self.training:
            mean = torch.stack(tuple(o.mean for o in out), dim=0)
            scale_tril = torch.stack(tuple(o.scale_tril for o in out), dim=0)
            return mean, scale_tril
        else:
            mean = torch.stack(tuple(o.mean for o in out), dim=-1)

            # Sometimes, gpytorch returns negative variances due to numerical errors.
            # Hence, clamp the output variance to the noise of the likelihood.
            stddev = torch.stack(
                tuple(torch.sqrt(o.variance.clamp(l.noise.item() ** 2, float('inf')))
                      for o, l in zip(out, self.likelihood)), dim=-1)
            return mean, torch.diag_embed(stddev)

    def add_data(self, state, action, next_state):
        """Add new data to GP-Model, independently to each GP."""
        new_x, new_y = self.state_actions_to_train_data(state, action, next_state)
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
                gp, self.max_num_points,
                weight_function=self._transform_weight_function(weight_function))
            print(len(gp.train_targets))

    def _transform_weight_function(self, weight_function=None):
        """Transform weight function according to input transform of the GP."""
        if not weight_function:
            return None

        if self.input_transform is None:
            def _wf(x):
                s = x[:, :-self.dim_action]
                a = x[:, -self.dim_action:]
                return weight_function(s, a)
        else:
            def _wf(x):
                s = self.input_transform.inverse(x[:, :-self.dim_action])
                a = x[:, -self.dim_action:]
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
    def state_actions_to_train_data(self, state, action, next_state):
        """Convert transition data to the gpytorch format.

        Parameters
        ----------
        state : torch.Tensor
            [N x d_x]
        action : torch.Tensor
            [N x d_u]
        next_state : torch.Tensor
            [N x d_x]

        Returns
        -------
        train_x : torch.Tensor
            [N x (d_x + d_u)]
        train_y : torch.Tensor
            [d_x x N], contiguous array
        """
        assert next_state.shape[-1] == state.shape[-1]
        train_x = self.state_actions_to_input_data(state, action)

        train_y = next_state.t().contiguous()

        return train_x, train_y


class RandomFeatureGPModel(ExactGPModel):
    """GP Model approximated by Random Fourier Features."""

    def __init__(self, state, action, next_state, num_features=1024,
                 mean=None, kernel=None, input_transform=None, max_num_points=None):
        super().__init__(state, action, next_state, mean, kernel, input_transform,
                         max_num_points)
        gps = []
        train_x, train_y = self.state_actions_to_train_data(state, action, next_state)
        for train_y_i, likelihood in zip(train_y, self.likelihood):
            gp = RandomFeatureGP(train_x, train_y_i, likelihood,
                                 num_features=num_features,
                                 mean=mean, kernel=kernel)
            gps.append(gp)
        self.gp = torch.nn.ModuleList(gps)


class SparseGPModel(ExactGPModel):
    """Sparse approximation of Exact GP models."""

    def __init__(self, state, action, next_state, inducing_points=None,
                 approximation='DTC', mean=None, kernel=None, input_transform=None,
                 max_num_points=None, max_inducing_points=None):
        super().__init__(state, action, next_state, mean, kernel, input_transform,
                         max_num_points)
        gps = []
        train_x, train_y = self.state_actions_to_train_data(state, action, next_state)
        if inducing_points is None:
            inducing_points = train_x

        self.max_inducing_points = max_inducing_points

        for train_y_i, likelihood in zip(train_y, self.likelihood):
            gp = SparseGP(train_x, train_y_i, likelihood, inducing_points,
                          approximation=approximation, mean=mean, kernel=kernel)
            gps.append(gp)
        self.gp = torch.nn.ModuleList(gps)

    def add_data(self, state, action, next_state, weight_function=None):
        """Add Data to GP and Re-Sparsify."""
        new_x, new_y = self.state_actions_to_train_data(state, action, next_state)
        for i, new_y_i in enumerate(new_y):
            arm_set = torch.cat((new_x, self.gp[i].train_inputs[0]), dim=0)
            inducing_points = bkb(self.gp[i], arm_set)
            print(len(arm_set), len(inducing_points))
            self.gp[i].set_inducing_points(inducing_points)
            add_data_to_gp(self.gp[i], new_x, new_y_i)
