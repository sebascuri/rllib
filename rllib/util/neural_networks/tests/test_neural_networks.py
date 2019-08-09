import torch
import torch.distributions
import torch.testing
import pytest
from rllib.util.neural_networks import *


@pytest.fixture(params=[None, [], [32], [64, 32]])
def layers(request):
    return request.param


@pytest.fixture(params=[None, 1, 32])
def batch_size(request):
    return request.param


@pytest.fixture(params=[1, 16])
def in_dim(request):
    return request.param


@pytest.fixture(params=[2, 4])
def out_dim(request):
    return request.param


@pytest.fixture(params=[0.1, 1.0, 10.0])
def temperature(request):
    return request.param


@pytest.fixture(params=[2, 5, 32])
def num_heads(request):
    return request.param


class TestDeterministicNN(object):
    @pytest.fixture(scope="class")
    def net(self):
        return DeterministicNN

    def test_output_shape(self, net, in_dim, out_dim, layers, batch_size):
        net = net(in_dim, out_dim, layers)
        if batch_size is None:
            t = torch.randn(in_dim)
            o = net(t)
            assert o.shape == torch.Size([out_dim])
        else:
            t = torch.randn(batch_size, in_dim)
            o = net(t)
            assert o.shape == torch.Size([batch_size, out_dim])

    def test_layers(self, net, in_dim, out_dim, layers):
        net = net(in_dim, out_dim, layers)
        layers = layers or list()

        # Check property assignment
        assert net.layers == layers

        # Check nn.parameters (+1: head)
        assert 2 * (len(layers) + 1) == len([*net.parameters()])

        # Check shapes
        layers.append(out_dim)
        for i, param in enumerate(net.parameters()):
            assert param.shape[0] == layers[i // 2]


class TestHeteroGaussianNN(object):
    @pytest.fixture(scope="class")
    def net(self):
        return HeteroGaussianNN

    def test_output_shape(self, net, in_dim, out_dim, layers, batch_size):
        net = net(in_dim, out_dim, layers)
        if batch_size is None:
            t = torch.randn(in_dim)
            o = net(t).sample()
            assert o.shape == torch.Size([out_dim])
        else:
            t = torch.randn(batch_size, in_dim)
            o = net(t).sample()
            assert o.shape == torch.Size([batch_size, out_dim])

    def test_output_properties(self, net, in_dim, out_dim, batch_size):
        net = net(in_dim, out_dim)
        if batch_size is None:
            t = torch.randn(in_dim)
        else:
            t = torch.randn(batch_size, in_dim)

        o = net(t)
        assert type(o) is torch.distributions.MultivariateNormal
        assert o.has_rsample
        assert not o.has_enumerate_support
        assert o.batch_shape == torch.Size([batch_size] if batch_size is not None else [])

    def test_temperature(self, net, temperature):
        net = net(4, 2, temperature=temperature)
        assert net.temperature == temperature

    def test_layers(self, net, in_dim, out_dim, layers):
        net = net(in_dim, out_dim, layers)
        layers = layers or list()

        # Check property assignment
        assert net.layers == layers

        # Check nn.parameters (+2: mean and covariance)
        assert 2 * (len(layers) + 2) == len([*net.parameters()])

        # Check shapes
        layers.append(out_dim)
        layers.append(out_dim)
        for i, param in enumerate(net.parameters()):
            assert param.shape[0] == layers[i // 2]


class TestHomoGaussianNN(object):
    @pytest.fixture(scope="class")
    def net(self):
        return HomoGaussianNN

    def test_output_shape(self, net, in_dim, out_dim, layers, batch_size):
        net = net(in_dim, out_dim, layers)
        if batch_size is None:
            t = torch.randn(in_dim)
            o = net(t).sample()
            assert o.shape == torch.Size([out_dim])
        else:
            t = torch.randn(batch_size, in_dim)
            o = net(t).sample()
            assert o.shape == torch.Size([batch_size, out_dim])

    def test_output_properties(self, net, in_dim, out_dim, batch_size):
        net = net(in_dim, out_dim)
        if batch_size is None:
            t = torch.randn(in_dim)
        else:
            t = torch.randn(batch_size, in_dim)

        o = net(t)
        assert type(o) is torch.distributions.MultivariateNormal
        assert o.has_rsample
        assert not o.has_enumerate_support
        assert o.batch_shape == torch.Size([batch_size] if batch_size is not None else [])

    def test_temperature(self, net, temperature):
        net = net(4, 2, temperature=temperature)
        assert net.temperature == temperature

    def test_layers(self, net, in_dim, out_dim, layers):
        net = net(in_dim, out_dim, layers)
        layers = layers or list()

        # Check property assignment
        assert net.layers == layers

        # Check nn.parameters (+1: mean and covariance has only 1 param)
        assert 2 * (len(layers) + 1) + 1 == len([*net.parameters()])

        # Check shapes
        layers.append(out_dim)
        i = 0
        for name, param in net.named_parameters():
            if name.startswith('_covariance'):
                assert param.shape[0] == out_dim
            else:
                assert param.shape[0] == layers[i // 2]
                i += 1


class TestCategoricalNN(object):
    @pytest.fixture(scope="class")
    def net(self):
        return CategoricalNN

    def test_output_shape(self, net, in_dim, out_dim, layers, batch_size):
        net = net(in_dim, out_dim, layers)
        if batch_size is None:
            t = torch.randn(in_dim)
            o = net(t).sample()
            assert o.shape == torch.Size([])
        else:
            t = torch.randn(batch_size, in_dim)
            o = net(t).sample()
            assert o.shape == torch.Size([batch_size])

    def test_output_properties(self, net, in_dim, out_dim, batch_size):
        net = net(in_dim, out_dim)
        if batch_size is None:
            t = torch.randn(in_dim)
        else:
            t = torch.randn(batch_size, in_dim)

        o = net(t)
        assert type(o) is torch.distributions.Categorical
        assert not o.has_rsample
        assert o.has_enumerate_support
        assert o.batch_shape == torch.Size([batch_size] if batch_size is not None else [])

    def test_temperature(self, net, temperature):
        net = net(4, 2, temperature=temperature)
        assert net.temperature == temperature

    def test_layers(self, net, in_dim, out_dim, layers):
        net = net(in_dim, out_dim, layers)
        layers = layers or list()

        # Check property assignment
        assert net.layers == layers

        # Check nn.parameters (+1: head)
        assert 2 * (len(layers) + 1) == len([*net.parameters()])

        # Check shapes
        layers.append(out_dim)
        for i, param in enumerate(net.parameters()):
            assert param.shape[0] == layers[i // 2]


class TestEnsembleNN(object):
    @pytest.fixture(scope="class")
    def net(self):
        return EnsembleNN

    def test_temperature(self, net, temperature):
        net = net(4, 2, temperature=temperature)
        assert net.temperature == temperature

    def test_num_heads(self, net, num_heads):
        net = net(4, 2, num_heads=num_heads)
        assert net.num_heads == num_heads

    def test_output_shape(self, net, out_dim, layers, num_heads, batch_size):
        in_dim = 4
        net = net(in_dim, out_dim, layers, num_heads=num_heads)
        if batch_size is None:
            t = torch.randn(in_dim)
            o = net(t).sample()
            assert o.shape == torch.Size([out_dim])
        else:
            t = torch.randn(batch_size, in_dim)
            o = net(t).sample()
            assert o.shape == torch.Size([batch_size, out_dim])

    def test_output_properties(self, net, out_dim, num_heads, batch_size):
        in_dim = 4
        net = net(in_dim, out_dim, num_heads=num_heads)
        if batch_size is None:
            t = torch.randn(in_dim)
        else:
            t = torch.randn(batch_size, in_dim)

        o = net(t)
        assert type(o) is torch.distributions.MultivariateNormal
        assert o.has_rsample
        assert not o.has_enumerate_support
        assert o.batch_shape == torch.Size([batch_size] if batch_size is not None else [])

    def test_layers(self, net, out_dim, num_heads, layers):
        in_dim = 4
        net = net(in_dim, out_dim, layers, num_heads=num_heads)
        layers = layers or list()

        # Check property assignment
        assert net.layers == layers

        # Check nn.parameters (+1: head)
        assert 2 * (len(layers) + 1) == len([*net.parameters()])

        # Check shapes
        layers.append(out_dim * num_heads)
        for i, (name, param) in enumerate(net.named_parameters()):
            assert param.shape[0] == layers[i // 2]


class TestFelixNet(object):
    @pytest.fixture(scope="class")
    def net(self):
        return FelixNet

    def test_output_shape(self, net, in_dim, out_dim, batch_size):
        net = net(in_dim, out_dim)
        if batch_size is None:
            t = torch.randn(in_dim)
            o = net(t).sample()
            assert o.shape == torch.Size([out_dim])
        else:
            t = torch.randn(batch_size, in_dim)
            o = net(t).sample()
            assert o.shape == torch.Size([batch_size, out_dim])

    def test_output_properties(self, net, in_dim, out_dim, batch_size):
        net = net(in_dim, out_dim)
        if batch_size is None:
            t = torch.randn(in_dim)
        else:
            t = torch.randn(batch_size, in_dim)

        o = net(t)
        assert type(o) is torch.distributions.MultivariateNormal
        assert o.has_rsample
        assert not o.has_enumerate_support
        assert o.batch_shape == torch.Size([batch_size] if batch_size is not None else [])

    def test_temperature(self, net, temperature):
        net = net(4, 2, temperature=temperature)
        assert net.temperature == temperature

    def test_layers(self, net, in_dim, out_dim, layers):
        net = net(in_dim, out_dim)
        layers = [64, 64]

        # Check property assignment
        assert net.layers == layers

        # Check nn.parameters (+2: mean and covariance have only weights)
        assert 2 * (len(layers)) + 2 == len([*net.parameters()])

        # Check shapes
        layers.append(out_dim)
        for i, param in enumerate(net.parameters()):
            assert param.shape[0] == layers[i // 2]

