"""Implementation of Gaussian Processes State-Space Models."""
import gpytorch
import torch
import torch.jit
import torch.nn

from rllib.util.gaussian_processes.exact_gp import ExactGP, SparseGP, RFF
from rllib.util.gaussian_processes.utilities import add_data_to_gp, bkb
from .abstract_model import AbstractModel


class ExactGPModel(AbstractModel):
    """An Exact GP State Space Model."""

    def __init__(self, state, action, next_state, mean=None, kernel=None,
                 approximation=None, input_transform=None):
        dim_state = state.shape[-1]
        dim_action = action.shape[-1]
        super().__init__(dim_state, dim_action)
        self.input_transform = input_transform
        train_x, train_y = self.state_actions_to_train_data(state, action, next_state)

        likelihoods = []
        gps = []
        for train_y_i in train_y:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            if approximation == 'sparse':
                gp = SparseGP(train_x, train_y_i, likelihood, train_x, mean, kernel)
            elif approximation == 'rff':
                gp = RFF(train_x, train_y_i, likelihood, num_features=1000, mean=mean,
                         kernel=kernel, outputscale=1., lengthscale=1.)
            else:
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
            stddev = torch.stack(tuple(o.stddev for o in out), dim=-1)
            return mean, torch.diag_embed(stddev)

    def add_data(self, state, action, next_state, weight_function=None):
        """Add new data to GP-Model, independently to each GP."""
        new_x, new_y = self.state_actions_to_train_data(state, action, next_state)
        for i, new_y_i in enumerate(new_y):
            add_data_to_gp(self.gp[i], new_x, new_y_i)

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


class SparseGreedyGPModel(ExactGPModel):
    r"""Sparse GP Model.

    Inducing points are selected by greedily maximizing
    ..math:: log det(1 + \lambda K(x, x))

    until a set of size `max_points' is built.

    References
    ----------
    Seeger, M., Williams, C., & Lawrence, N. (2003).
    Fast forward selection to speed up sparse Gaussian process regression.

    Gomes, R., & Krause, A. (2010).
    Budgeted Nonparametric Learning from Data Streams. ICML.
    """

    def __init__(self, state, action, next_state, max_points, mean=None, kernel=None,
                 approximation='sparse', input_transform=None):
        super().__init__(state, action, next_state, mean, kernel, approximation,
                         input_transform)
        self.max_points = max_points

    def add_data(self, state, action, next_state, weight_function=None):
        """Add new data to GP-Model, independently to each GP."""
        new_x, new_y = self.state_actions_to_train_data(state, action, next_state)
        for i, new_y_i in enumerate(new_y):
            add_data_to_gp(self.gp[i], new_x, new_y_i, max_num=self.max_points)


class SparseWeightedGreedyGPModel(ExactGPModel):
    r"""Sparse GP Model.

    Inducing points are selected by greedily maximizing
    ..math:: log det(1 + \lambda K(x, x)) J(x)

    until a set of size `max_points' is built.

    References
    ----------
    McIntire, M., Ratner, D., & Ermon, S. (2016).
    Sparse Gaussian Processes for Bayesian Optimization. UAI.
    """

    def __init__(self, state, action, next_state, max_points, mean=None, kernel=None,
                 approximation='sparse', input_transform=None):
        super().__init__(state, action, next_state, mean, kernel, approximation,
                         input_transform)
        self.max_points = max_points

    def add_data(self, state, action, next_state, weight_function=None):
        """Add new data to GP-Model, independently to each GP."""
        new_x, new_y = self.state_actions_to_train_data(state, action, next_state)
        for i, new_y_i in enumerate(new_y):

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

            add_data_to_gp(self.gp[i], new_x, new_y_i, max_num=self.max_points,
                           weight_function=_wf)


class BKBGPModel(ExactGPModel):
    r"""Sparse GP Model.

    Inducing points are selected by sampling according to
    ..math:: i \sim Bernoulli(\bar{q} \sigma^2(x)),
    where \sigma^2 is the predictive variance of the current GP model and \bar{q} a
    parameter of the algorithm.

    References
    ----------
    McIntire, M., Ratner, D., & Ermon, S. (2016).
    Sparse Gaussian Processes for Bayesian Optimization. UAI.
    """

    def __init__(self, state, action, next_state, q_bar, mean=None, kernel=None,
                 approximation='sparse', input_transform=None):
        super().__init__(state, action, next_state, mean, kernel, approximation,
                         input_transform)
        self.q_bar = q_bar

    def add_data(self, state, action, next_state, weight_function=None):
        """Add new data to GP-Model, independently to each GP."""
        new_x, new_y = self.state_actions_to_train_data(state, action, next_state)
        for i, new_y_i in enumerate(new_y):
            bkb(self.gp[i], new_x, new_y_i, self.q_bar)
