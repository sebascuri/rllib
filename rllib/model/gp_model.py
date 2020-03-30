"""Implementation of Gaussian Processes State-Space Models."""
import gpytorch
import torch
import torch.jit
import torch.nn

from rllib.util.gaussian_processes.exact_gp import ExactGP  # , MultitaskExactGP
from rllib.util.gaussian_processes.utilities import add_data_to_gp, plot_gp_inputs
from .abstract_model import AbstractModel

import matplotlib.pyplot as plt


class ExactGPModel(AbstractModel):
    """An Exact GP State Space Model."""

    def __init__(self, state, action, next_state, mean=None, kernel=None,
                 input_transform=None, max_num_points=200):
        dim_state = state.shape[-1]
        dim_action = action.shape[-1]
        super().__init__(dim_state, dim_action)
        self.input_transform = input_transform
        train_x, train_y = self.state_actions_to_train_data(state, action, next_state)
        # if dim_state == 1:
        likelihoods = []
        gps = []
        for train_y_i in train_y:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            gps.append(ExactGP(train_x, train_y_i, likelihood, mean, kernel))
            likelihoods.append(likelihood)

        self.likelihood = torch.nn.ModuleList(likelihoods)
        self.gp = torch.nn.ModuleList(gps)

        self.max_num_points = max_num_points

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

    def add_data(self, state, action, next_state):
        """Add new data to GP-Model, independently to each GP."""
        train_x, train_y = self.state_actions_to_train_data(state, action, next_state)
        for i, train_y_i in enumerate(train_y):
            add_data_to_gp(self.gp[i], train_x, train_y_i, max_num=self.max_num_points)

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
        train_x = torch.cat((self.input_transform(state), action), dim=-1)
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

    def plot_inputs(self):
        """Plot GP inputs."""
        fig, axes = plt.subplots(self.dim_state, 1, sharex='row')
        for i, model in enumerate(self.gp):
            plot_gp_inputs(model, axes[i])

            axes[i].set_xlim([-180, 180])
            axes[i].set_ylim([-15, 15])
            axes[i].set_ylabel('Angular Velocity')
        axes[0].set_title('Input Data of GP.')
        axes[-1].set_xlabel('Angle')
        plt.show()
