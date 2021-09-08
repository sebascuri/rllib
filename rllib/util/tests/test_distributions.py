import torch
import torch.testing

from rllib.util.distributions import Delta


class TestDelta(object):
    def test_correctness(self):
        x = torch.randn(32, 4)
        dist = Delta(v=x, event_dim=1)

        s = dist.sample((10,))
        torch.testing.assert_allclose(s, x.expand(10, 32, 4))

    def test_entropy(self):
        x = torch.randn(32, 4)
        dist = Delta(v=x)
        torch.testing.assert_allclose(dist.entropy(), torch.zeros(32, 4))
        torch.testing.assert_allclose(dist.variance, torch.zeros(32, 4))

        dist = Delta(v=x, event_dim=1)
        torch.testing.assert_allclose(dist.entropy(), torch.zeros(32))
        torch.testing.assert_allclose(dist.variance, torch.zeros(32, 4))

    def test_shapes(self):
        dist = Delta(v=torch.randn(32, 4), event_dim=1)
        assert dist.batch_shape == torch.Size([32])
        assert dist.event_shape == torch.Size([4])
        assert dist.entropy().shape == torch.Size([32])
        assert dist.variance.shape == torch.Size([32, 4])

        dist = Delta(v=torch.randn(32, 4))
        assert dist.batch_shape == torch.Size([32, 4])
        assert dist.event_shape == torch.Size([])
        assert dist.entropy().shape == torch.Size([32, 4])
        assert dist.variance.shape == torch.Size([32, 4])

        dist = Delta(v=torch.randn(16), event_dim=1)
        assert dist.batch_shape == torch.Size([])
        assert dist.event_shape == torch.Size([16])
        assert dist.entropy().shape == torch.Size([])
        assert dist.variance.shape == torch.Size([16])

        dist = Delta(v=torch.randn(16))
        assert dist.batch_shape == torch.Size([16])
        assert dist.event_shape == torch.Size([])
        assert dist.entropy().shape == torch.Size([16])
        assert dist.variance.shape == torch.Size([16])
