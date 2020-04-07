"""Exact GP Model."""
import gpytorch
from gpytorch.models.exact_prediction_strategies import prediction_strategy

import torch
from torch.distributions import MultivariateNormal
import numpy as np


class RFF(gpytorch.models.ExactGP):
    """RFF approximation to a GP."""

    def __init__(self, train_x, train_y, likelihood, num_features, mean=None,
                 kernel=None, outputscale=1., lengthscale=1.):
        super().__init__(train_x, train_y, likelihood)
        self.num_features = num_features
        self.dim = train_x.shape[-1]

        self.w = lengthscale * torch.randn(self.dim, self.num_features)
        self.b = 2 * np.pi * torch.rand(1, num_features)
        self._output_scale = outputscale
        self._length_scale = lengthscale

        # self.layer = nn.Linear()
        if mean is None:
            mean = gpytorch.means.ZeroMean()
        self.mean_module = mean

        self.jitter = 1e-4
        self.update_cache()

    def set_train_data(self, inputs=None, targets=None, strict=True):
        """Set raining data and update cache."""
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            inputs = tuple(
                input_.unsqueeze(-1) if input_.ndimension() == 1 else input_ for input_
                in inputs)
            self.train_inputs = inputs
        if targets is not None:
            self.train_targets = targets
        self.update_cache()

    def update_cache(self):
        """Updated cached data."""
        with torch.no_grad():
            prior_mean, prior_covar = self._mean_covar(self.train_inputs[0])

            prior_features = self.features(self.train_inputs[0])
            residual = self.train_targets - prior_mean
            l = torch.cholesky(prior_covar)
            kf_kff_inv = torch.cholesky_solve(prior_features, l).transpose(-2, -1)

            self.kf_kff_inv_y = kf_kff_inv @ residual
            self.qff = kf_kff_inv @ prior_features

    @property
    def output_scale(self):
        """Get output scale."""
        return self._output_scale

    @output_scale.setter
    def output_scale(self, new_output_scale):
        """Set output scale."""
        self._output_scale = new_output_scale

    @property
    def length_scale(self):
        """Get length scale."""
        return self._length_scale

    @length_scale.setter
    def length_scale(self, new_length_scale):
        """Set length scale."""
        self._length_scale = new_length_scale
        self.w = new_length_scale * torch.randn(self.dim, self.num_features)

    def features(self, x):
        """Compute features from RFF."""
        scale = torch.sqrt(torch.tensor(2. / self.num_features) * self.output_scale)
        return scale * torch.cos(x @ self.w + self.b)

    def _mean_covar(self, x):
        """Calculate mean and covariance at location x."""
        rff = self.features(x)
        kernel = rff @ rff.transpose(-2, -1)
        kernel += torch.eye(kernel.shape[-1]) * self.jitter
        return self.mean_module(x), kernel

    def __call__(self, *args, **kwargs):
        test_mean, test_cov = self._mean_covar(args[0])
        test_features = self.features(args[0])

        posterior_mean = test_mean + test_features @ self.kf_kff_inv_y
        posterior_cov = test_cov - test_features @ self.qff @ test_features.transpose(
            -1, -2)
        return gpytorch.distributions.MultivariateNormal(posterior_mean, posterior_cov)

    def forward(self, new_x):
        """Compute GP distribution at locations new_x."""
        mean, covar = self._mean_covar(new_x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class SparseGP(gpytorch.models.GP):
    """Sparse GP Models."""

    def __init__(self, train_x, train_y, likelihood, inducing_points,
                 mean=None, kernel=None, approximation='DTC'):
        super().__init__()
        self.train_inputs = (train_x,)
        self.train_targets = train_y
        self.xu = inducing_points

        self.likelihood = likelihood
        self.approximation = approximation

        if mean is None:
            mean = gpytorch.means.ZeroMean()
        self.mean_module = mean

        if kernel is None:
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = kernel

        self.jitter = 1e-3
        self.update_cache()

    @property
    def output_scale(self):
        """Get output scale."""
        return self.covar_module.outputscale

    @output_scale.setter
    def output_scale(self, new_output_scale):
        """Set output scale."""
        self.covar_module.outputscale = new_output_scale

    @property
    def length_scale(self):
        """Get length scale."""
        return self.covar_module.base_kernel.lengthscale

    @length_scale.setter
    def length_scale(self, new_length_scale):
        """Set length scale."""
        self.covar_module.base_kernel.lengthscale = new_length_scale

    def set_train_data(self, inputs, targets, strict=False):
        """Set new training data."""
        self.train_inputs = (inputs,)
        self.train_targets = targets

    def set_inducing_points(self, inducing_points):
        """Set new inducing points."""
        self.xu = inducing_points
        self.update_cache()

    def update_cache(self):
        """Update cached quantities."""
        n = self.train_inputs[0].shape[0]
        m = self.xu.shape[0]

        k_uu = self.covar_module(self.xu).add_jitter(self.jitter).evaluate()
        self.l_uu = k_uu.cholesky()

        k_uf = self.covar_module(self.xu, self.train_inputs[0]).evaluate()

        w = k_uf.triangular_solve(self.l_uu, upper=False)[0]
        diag = self.likelihood.noise * torch.ones(n)
        if self.approximation == "FITC":
            k_ff_diag = self.covar_module(self.train_inputs[0], diag=True).evaluate()
            q_ff_diag = w.pow(2).sum(dim=0)
            diag = diag + k_ff_diag - q_ff_diag

        w_d_inv = w / diag
        k = (w_d_inv @ w.transpose(-2, -1)).contiguous()
        k.view(-1)[::m + 1] += 1  # add identity matrix to K
        self.l = k.cholesky()

        # get y_residual and convert it into 2D tensor for packing
        y_residual = self.train_targets - self.mean_module(self.train_inputs[0])
        y_2d = y_residual.reshape(-1, n).transpose(-2, -1)
        self.w_d_inv_y = w_d_inv @ y_2d

    def forward(self, x_new):
        """Compute distribution."""
        k_us = self.covar_module(self.xu, x_new).evaluate()
        w_s = k_us.triangular_solve(self.l_uu, upper=False)[0]

        if x_new.dim() == 2:
            w_d_inv_y = self.w_d_inv_y
        else:
            w_d_inv_y = self.w_d_inv_y.expand(x_new.shape[0], *self.w_d_inv_y.shape)
        pack = torch.cat((w_d_inv_y, w_s), dim=-1)
        l_inv_pack = pack.triangular_solve(self.l, upper=False)[0]

        # unpack
        l_inv_w_d_inv_y = l_inv_pack[..., :self.w_d_inv_y.shape[-1]]
        l_inv_w_s = l_inv_pack[..., self.w_d_inv_y.shape[-1]:]

        loc = l_inv_w_d_inv_y.transpose(-2, -1) @ l_inv_w_s
        loc = loc.squeeze(1)

        k_ss = self.covar_module(x_new).add_jitter(
            self.likelihood.noise.data.item() + self.jitter).evaluate()
        q_ss = w_s.transpose(-2, -1) @ w_s  # 8 x 8
        cov = k_ss - q_ss + l_inv_w_s.transpose(-2, -1) @ l_inv_w_s

        # cov_shape = self.train_targets.shape[:-1] + (c, c)
        # cov = cov.expand(cov_shape)

        return gpytorch.distributions.MultivariateNormal(
            loc + self.mean_module(x_new), cov)


class ExactGP(gpytorch.models.ExactGP):
    """Exact GP Model."""

    def __init__(self, train_x, train_y, likelihood, mean=None, kernel=None):
        super().__init__(train_x, train_y, likelihood)
        if mean is None:
            mean = gpytorch.means.ZeroMean()
        self.mean_module = mean

        if kernel is None:
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = kernel

    @property
    def output_scale(self):
        """Get output scale."""
        return self.covar_module.outputscale

    @output_scale.setter
    def output_scale(self, new_output_scale):
        """Set output scale."""
        self.covar_module.outputscale = new_output_scale

    @property
    def length_scale(self):
        """Get length scale."""
        return self.covar_module.base_kernel.lengthscale

    @length_scale.setter
    def length_scale(self, new_length_scale):
        """Set length scale."""
        self.covar_module.base_kernel.lengthscale = new_length_scale

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
