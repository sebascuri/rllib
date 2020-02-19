"""Implementation of a Variational GP with gpytorch."""

import gpytorch


class ApproximateGPModel(gpytorch.models.ApproximateGP):
    """Approximate GP model."""

    def __init__(self, inducing_points, learn_loc=True):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-1))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=learn_loc
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        """Forward computation of GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
