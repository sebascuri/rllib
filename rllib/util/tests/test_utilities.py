import numpy as np
import pytest
import torch
import torch.testing
from torch.distributions import Categorical, MultivariateNormal, kl_divergence

from rllib.util.distributions import Delta
from rllib.util.utilities import (
    get_backend,
    integrate,
    mellow_max,
    separated_kl,
    tensor_to_distribution,
)


class TestGetBackend(object):
    def test_correctness(self):
        assert get_backend(torch.randn(4)) is torch
        assert get_backend(np.random.randn(3)) is np

        with pytest.raises(TypeError):
            get_backend([1, 2, 3])


class TestIntegrate(object):
    def test_discrete_distribution(self):
        d = Categorical(torch.tensor([0.1, 0.2, 0.3, 0.4]))

        def _function(a):
            return 2 * a

        torch.testing.assert_allclose(integrate(_function, d), 4.0)

    def test_delta(self):
        d = Delta(v=torch.tensor([0.2]))

        def _function(a):
            return 2 * a

        torch.testing.assert_allclose(
            integrate(_function, d, num_samples=10), torch.tensor([0.4])
        )

    def test_multivariate_normal(self):
        d = MultivariateNormal(torch.tensor([0.2]), scale_tril=1e-6 * torch.eye(1))

        def _function(a):
            return 2 * a

        torch.testing.assert_allclose(
            integrate(_function, d, num_samples=100), 0.4, rtol=1e-3, atol=1e-3
        )


class TestMellowMax(object):
    @pytest.fixture(params=[0.1, 1, 10], scope="class")
    def omega(self, request):
        return request.param

    @pytest.fixture(params=[None, 1, 4], scope="class")
    def batch_size(self, request):
        return request.param

    @pytest.fixture(params=[1, 2], scope="class")
    def dim(self, request):
        return request.param

    def test_correctness(self, omega, batch_size, dim):
        if batch_size:
            x = torch.randn(batch_size, dim)
        else:
            x = torch.randn(dim)

        mm = mellow_max(x, omega)

        torch.testing.assert_allclose(
            mm, 1 / omega * torch.log(torch.mean(torch.exp(omega * x), dim=-1))
        )
        if batch_size:
            torch.testing.assert_allclose(mm.shape, torch.Size([batch_size]))
        else:
            torch.testing.assert_allclose(mm.shape, torch.Size([]))


class TestTensor2Dist(object):
    @pytest.fixture(params=[None, 1, 4], scope="class")
    def batch_size(self, request):
        return request.param

    @pytest.fixture(params=[1, 2], scope="class")
    def dim(self, request):
        return request.param

    def test_categorical(self, batch_size, dim):
        if batch_size:
            x = torch.randn(batch_size, dim)
        else:
            x = torch.randn(dim)

        d = tensor_to_distribution(x)
        logits = x - x.logsumexp(dim=-1, keepdim=True)
        assert isinstance(d, Categorical)
        torch.testing.assert_allclose(d.logits, logits)
        torch.testing.assert_allclose(d.probs, torch.softmax(x, dim=-1))

        assert isinstance(d.sample(), torch.LongTensor)
        assert d.sample().shape == x.shape[:-1]

    def test_delta(self, batch_size, dim):
        if batch_size:
            mu = torch.randn(batch_size, dim)
            scale = 0 * torch.eye(dim).expand(batch_size, dim, dim)
        else:
            mu = torch.randn(dim)
            scale = 0 * torch.eye(dim)

        d = tensor_to_distribution((mu, scale))
        assert isinstance(d, Delta)
        torch.testing.assert_allclose(d.mean, mu)

        assert d.sample().dtype is torch.get_default_dtype()
        assert d.sample().shape == mu.shape

    def test_multivariate_normal(self, batch_size, dim):
        if batch_size:
            mu = torch.randn(batch_size, dim)
            scale = 0.1 * torch.eye(dim).expand(batch_size, dim, dim)
        else:
            mu = torch.randn(dim)
            scale = 0.1 * torch.eye(dim)

        d = tensor_to_distribution((mu, scale))
        assert isinstance(d, MultivariateNormal)
        torch.testing.assert_allclose(d.mean, mu)
        torch.testing.assert_allclose(d.covariance_matrix, scale ** 2)

        assert d.sample().dtype is torch.get_default_dtype()
        assert d.sample().shape == mu.shape


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
            lower[..., torch.arange(dim), torch.arange(dim)]
        )
        if batch_size:
            lower += torch.eye(dim).repeat(batch_size, 1, 1)
        else:
            lower += torch.eye(dim)

        return MultivariateNormal(m, scale_tril=lower)

    def test_separated_kl(self, dim, batch_size):
        p = self.get_multivariate_normal(dim, batch_size)
        q = self.get_multivariate_normal(dim, batch_size)

        torch.testing.assert_allclose(
            kl_divergence(p, q).sum(), sum(separated_kl(p, q)).sum()
        )

        torch.testing.assert_allclose(
            kl_divergence(q, p).sum(), sum(separated_kl(q, p)).sum()
        )

    def test_correctness(self, dim, batch_size):
        p = self.get_multivariate_normal(dim, batch_size)
        q = self.get_multivariate_normal(dim, batch_size)
        kl_mean, kl_var = separated_kl(p, q)
        if batch_size is None:
            kl_mean_ = (
                0.5
                * (q.loc - p.loc)
                @ torch.inverse(q.covariance_matrix)
                @ (q.loc - p.loc)
            ).mean()
        else:
            kl_mean_ = (
                0.5
                * (q.loc - p.loc).unsqueeze(1)
                @ torch.inverse(q.covariance_matrix)
                @ (q.loc - p.loc).unsqueeze(-1)
            ).mean()

        n = q.covariance_matrix.shape[-1]
        ratio = torch.inverse(q.covariance_matrix) @ p.covariance_matrix
        kl_var_ = (
            0.5
            * (
                torch.logdet(q.covariance_matrix)
                - torch.logdet(p.covariance_matrix)
                + ratio[..., torch.arange(n), torch.arange(n)].sum(-1)
                - n
            )
        ).mean()

        torch.testing.assert_allclose(kl_mean.mean(), kl_mean_)

        torch.testing.assert_allclose(kl_var.mean(), kl_var_)
