import numpy as np
import pytest
import scipy
import torch
import torch.testing
from torch.distributions import kl_divergence, MultivariateNormal

from rllib.util.utilities import discount_cumsum, separated_kl


class TestDiscountedCumSum(object):
    @pytest.fixture(params=[1, 0.99, 0.9, 0], scope="class")
    def gamma(self, request):
        return request.param

    @pytest.fixture(params=[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 2, 1, 0.2, 0.4]], scope="class")
    def rewards(self, request):
        return request.param

    def test_correctness(self, gamma):
        rewards = [1, 0.5, 2, -0.2]
        cum_rewards = [1 + 0.5 * gamma + 2 * gamma ** 2 - 0.2 * gamma ** 3,
                       0.5 + 2 * gamma - 0.2 * gamma ** 2,
                       2 - 0.2 * gamma,
                       -0.2
                       ]
        scipy.allclose(cum_rewards, discount_cumsum(np.array(rewards), gamma))

        torch.testing.assert_allclose(cum_rewards,
                                      discount_cumsum(torch.tensor(rewards), gamma))

    def test_shape_and_type(self, rewards, gamma):
        np_returns = discount_cumsum(np.array(rewards), gamma)
        assert np_returns.shape == (len(rewards),)
        assert type(np_returns) is np.ndarray

        t_returns = discount_cumsum(
            torch.tensor(rewards, dtype=torch.get_default_dtype()), gamma)
        assert t_returns.shape == torch.Size((len(rewards),))
        assert type(t_returns) is torch.Tensor

        torch.testing.assert_allclose(t_returns, np_returns)


class TestSeparatedKL(object):
    @pytest.fixture(params=[1, 10], scope="class")
    def dim(self, request):
        return request.param

    @pytest.fixture(params=[1, 5, None], scope="class")
    def batch_size(self, request):
        return request.param

    def get_multivariate_normal(self, dim, batch_size):
        if batch_size:
            m = torch.randn(batch_size, dim)
            lower = torch.tril(torch.randn(batch_size, dim, dim))
        else:
            m = torch.randn(dim)
            lower = torch.tril(torch.randn(dim, dim))

        lower[..., torch.arange(dim), torch.arange(dim)] = torch.nn.functional.softplus(
            lower[..., torch.arange(dim), torch.arange(dim)])
        if batch_size:
            lower += torch.eye(dim).repeat(batch_size, 1, 1)
        else:
            lower += torch.eye(dim)

        return MultivariateNormal(m, scale_tril=lower)

    def test_separated_kl(self, dim, batch_size):
        p = self.get_multivariate_normal(dim, batch_size)
        q = self.get_multivariate_normal(dim, batch_size)

        torch.testing.assert_allclose(
            kl_divergence(p, q).mean(), sum(separated_kl(p, q))
        )

        torch.testing.assert_allclose(
            kl_divergence(q, p).mean(), sum(separated_kl(q, p))
        )
