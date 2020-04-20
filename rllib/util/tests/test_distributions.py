import numpy as np
import pytest
import scipy
import torch
import torch.testing
from torch.distributions import MultivariateNormal, ComposeTransform, \
    TransformedDistribution
from torch.distributions.transforms import AffineTransform, SigmoidTransform
from rllib.util.distributions import Delta, TanhTransform


class TestTanhTransform(object):
    @staticmethod
    def get_distribution(dist_type):
        base_dist = MultivariateNormal(torch.zeros(3), 3 * torch.eye(3))

        if dist_type == 'base':
            return base_dist
        elif dist_type == 'tanh':
            tanh = TanhTransform()
            return TransformedDistribution(base_dist, tanh)
        elif dist_type == 'equiv':
            equiv_tanh = ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(),
                                           AffineTransform(-1., 2.)])
            return TransformedDistribution(base_dist, equiv_tanh)

    @pytest.fixture(scope="class", params=['base', 'tanh', 'equiv'])
    def distribution(self, request):
        return self.get_distribution(request.param)

    @pytest.mark.parametrize("distribution_type", ['base', 'tanh', 'equiv'])
    def test_range(self, distribution_type):
        distribution = self.get_distribution(distribution_type)
        x = distribution.rsample((10,))
        if isinstance(distribution, MultivariateNormal):
            x = x.tanh()

        assert torch.all(x.abs() <= 1)

    def test_shape(self, distribution):
        x = distribution.rsample((10,))
        if isinstance(distribution, MultivariateNormal):
            x = x.tanh()

        assert x.shape == torch.Size([10, 3])

    def test_log_prob(self, distribution):
        x = distribution.rsample((10,))
        if isinstance(distribution, MultivariateNormal):
            x = x.tanh()

        tanh_dist = self.get_distribution('tanh')
        etanh_dist = self.get_distribution('equiv')
        torch.testing.assert_allclose(tanh_dist.log_prob(x), etanh_dist.log_prob(x))


class TestDelta(object):
    def test_correctness(self):
        x = torch.randn(32, 4)
        dist = Delta(x, event_dim=1)

        s = dist.sample((10,))
        torch.testing.assert_allclose(s, x.expand(10, 32, 4))

    def test_entropy(self):
        x = torch.randn(32, 4)
        dist = Delta(x)
        torch.testing.assert_allclose(dist.entropy(), torch.zeros(32, 4))
        torch.testing.assert_allclose(dist.variance, torch.zeros(32, 4))

        dist = Delta(x, event_dim=1)
        torch.testing.assert_allclose(dist.entropy(), torch.zeros(32))
        torch.testing.assert_allclose(dist.variance, torch.zeros(32, 4))

    def test_shapes(self):
        dist = Delta(torch.randn(32, 4), event_dim=1)
        assert dist.batch_shape == torch.Size([32])
        assert dist.event_shape == torch.Size([4])
        assert dist.entropy().shape == torch.Size([32])
        assert dist.variance.shape == torch.Size([32, 4])

        dist = Delta(torch.randn(32, 4))   # event_dim = 0
        assert dist.batch_shape == torch.Size([32, 4])
        assert dist.event_shape == torch.Size([])
        assert dist.entropy().shape == torch.Size([32, 4])
        assert dist.variance.shape == torch.Size([32, 4])

        dist = Delta(torch.randn(16), event_dim=1)
        assert dist.batch_shape == torch.Size([])
        assert dist.event_shape == torch.Size([16])
        assert dist.entropy().shape == torch.Size([])
        assert dist.variance.shape == torch.Size([16])

        dist = Delta(torch.randn(16))   # event_dim = 0
        assert dist.batch_shape == torch.Size([16])
        assert dist.event_shape == torch.Size([])
        assert dist.entropy().shape == torch.Size([16])
        assert dist.variance.shape == torch.Size([16])
