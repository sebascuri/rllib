import numpy as np
import pytest
import scipy
import torch
import torch.testing
from torch.distributions import kl_divergence, MultivariateNormal, Categorical
from rllib.dataset.datatypes import RawObservation
from rllib.util.distributions import Delta
from rllib.util.utilities import discount_cumsum, discount_sum, separated_kl, \
    integrate, mellow_max, mc_return, get_backend, tensor_to_distribution


class TestGetBackend(object):
    def test_correctness(self):
        assert get_backend(torch.randn(4)) is torch
        assert get_backend(np.random.randn(3)) is np

        with pytest.raises(TypeError):
            get_backend([1, 2, 3])


class TestIntegrate(object):
    def test_discrete_distribution(self):
        d = Categorical(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        function = lambda a: 2 * a
        torch.testing.assert_allclose(integrate(function, d), 4.0)

    def test_delta(self):
        d = Delta(torch.tensor([0.2]))
        function = lambda a: 2 * a
        torch.testing.assert_allclose(integrate(function, d, num_samples=10), 0.4)

    def test_multivariate_normal(self):
        d = MultivariateNormal(torch.tensor([0.2]), scale_tril=1e-6 * torch.eye(1))
        function = lambda a: 2 * a
        torch.testing.assert_allclose(integrate(function, d, num_samples=100), 0.4,
                                      rtol=1e-3, atol=1e-3)


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
            mm, 1 / omega * torch.log(torch.mean(torch.exp(omega * x), dim=-1)))
        if batch_size:
            torch.testing.assert_allclose(mm.shape, torch.Size([batch_size, ]))
        else:
            torch.testing.assert_allclose(mm.shape, torch.Size([]))


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
        assert scipy.allclose(cum_rewards, discount_cumsum(np.array(rewards), gamma))

        torch.testing.assert_allclose(cum_rewards,
                                      discount_cumsum(torch.tensor(rewards), gamma))

        torch.testing.assert_allclose(cum_rewards[0],
                                      discount_sum(torch.tensor(rewards), gamma))

    def test_shape_and_type(self, rewards, gamma):
        np_returns = discount_cumsum(np.array(rewards), gamma)
        assert np_returns.shape == (len(rewards),)
        assert type(np_returns) is np.ndarray

        t_returns = discount_cumsum(
            torch.tensor(rewards, dtype=torch.get_default_dtype()), gamma)
        assert t_returns.shape == torch.Size((len(rewards),))
        assert type(t_returns) is torch.Tensor

        torch.testing.assert_allclose(t_returns, np_returns)


class TestMCReturn(object):
    @pytest.fixture(params=[1, 0.99, 0.9, 0], scope="class")
    def gamma(self, request):
        return request.param

    @pytest.fixture(params=[True, False], scope="class")
    def value_function(self, request):
        if request:
            return lambda x: 0.01
        else:
            return None

    @pytest.fixture(params=[1, 0.1, 0], scope="class")
    def entropy_reg(self, request):
        return request.param

    def test_correctness(self, gamma, value_function, entropy_reg):
        trajectory = [
            RawObservation(0, 0, reward=1, done=False, entropy=0.2).to_torch(),
            RawObservation(0, 0, reward=0.5, done=False, entropy=0.3).to_torch(),
            RawObservation(0, 0, reward=2, done=False, entropy=0.5).to_torch(),
            RawObservation(0, 0, reward=-0.2, done=False, entropy=-0.2).to_torch(),
        ]

        r0 = 1 + entropy_reg * 0.2
        r1 = 0.5 + entropy_reg * 0.3
        r2 = 2 + entropy_reg * 0.5
        r3 = -0.2 - entropy_reg * 0.2

        v = 0.01 if value_function is not None else 0

        reward = mc_return(trajectory, gamma, value_function, entropy_reg)

        torch.testing.assert_allclose(
            reward, r0 + r1 * gamma + r2 * gamma ** 2 + r3 * gamma ** 3 + v * gamma ** 4
        )
        assert mc_return([], gamma, value_function, entropy_reg) == 0


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
