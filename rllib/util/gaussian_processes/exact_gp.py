"""Exact GP Model."""
import gpytorch


class ExactGP(gpytorch.models.ExactGP):
    """Exact GP Model."""

    def __init__(self, train_x, train_y, likelihood, mean=None, kernel=None):
        super().__init__(train_x, train_y, likelihood)
        if mean is None:
            mean = gpytorch.means.ConstantMean()
        self.mean_module = mean
        if kernel is None:
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = kernel

    def forward(self, x):
        """Forward computation of GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultitaskExactGP(gpytorch.models.ExactGP):
    """Multitask Exact GP."""

    def __init__(self, train_x, train_y, likelihood, num_tasks=2,
                 mean=None, kernel=None):
        super().__init__(train_x, train_y, likelihood)
        if mean is None:
            mean = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(),
                                                num_tasks=num_tasks)
        self.mean_module = mean
        if kernel is None:
            kernel = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.RBFKernel(),
                                                      num_tasks=num_tasks, rank=1)
        self.covar_module = kernel

    def forward(self, x):
        """Forward computation of GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
