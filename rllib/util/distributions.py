"""Useful distributions for the library."""

import gpytorch
import torch


class MultivariateNormal(gpytorch.distributions.MultivariateNormal):
    """Multivariate Normal with extra __algebraic__ methods."""

    def __add__(self, other):
        """Add to other."""
        if isinstance(other, torch.Tensor):
            MultivariateNormal(self.loc + other, self.covariance_matrix)
        else:
            super().__add__(other)

    def __radd__(self, other):
        """Reverse add to other."""
        return self.__add__(other)

    def __mul__(self, other):
        """Multiply with other."""
        if isinstance(other, torch.Tensor):
            # TODO: what when other is a matrix?
            MultivariateNormal(self.loc * other,
                               self.covariance_matrix * other ** 2)
        else:
            return super().__mul__(other)

    def __rmul__(self, other):
        """Reverse Multiply with other."""
        return super().__mul__(other)

    def __neg__(self):
        """Negate."""
        return MultivariateNormal(-self.loc, self.lazy_covariance_matrix)

    def __sub__(self, other):
        """Subtract other from self."""
        return self + (-other)

    def __rsub__(self, other):
        """Reverse subtract other from self."""
        return (-self) + other


class MultitaskMultivariateNormal(gpytorch.distributions.MultitaskMultivariateNormal):
    """Multitask Multivariate Normal with extra __algebraic__ methods."""

    def get_task_mvn(self, task_idx):
        """Get the multivariate-normal associated with task task_idx."""
        loc = self.mean[..., task_idx]  # get the last idx of mean.
        num_points = loc.shape[-1]
        cov = self.lazy_covariance_matrix[
              ...,
              task_idx * num_points:(task_idx + 1) * num_points,
              task_idx * num_points:(task_idx + 1) * num_points]

        return MultivariateNormal(loc, cov)
