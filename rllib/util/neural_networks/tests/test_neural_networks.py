import pytest
import torch
import torch.distributions
import torch.jit
import torch.testing

from rllib.util.distributions import Delta
from rllib.util.neural_networks import (
    CategoricalNN,
    DeterministicNN,
    Ensemble,
    FelixNet,
    HeteroGaussianNN,
    HomoGaussianNN,
)
from rllib.util.neural_networks.utilities import count_vars
from rllib.util.utilities import tensor_to_distribution


@pytest.fixture(params=[[], [32], [64, 32]])
def layers(request):
    return request.param


@pytest.fixture(params=["ReLU", "tanh"])
def non_linearity(request):
    return request.param


@pytest.fixture(params=[None, 1, 32])
def batch_size(request):
    return request.param


@pytest.fixture(params=[1, 16])
def in_dim(request):
    return (request.param,)


@pytest.fixture(params=[2, 4])
def out_dim(request):
    return (request.param,)


@pytest.fixture(params=[5, 32])
def num_heads(request):
    return request.param


def _test_from_other(object_, class_):
    other = class_.from_other(object_, copy=False)

    assert isinstance(other, class_)
    assert other is not object_

    try:
        other = torch.jit.script(other)
    except:  # noqa: E722
        pass
    other_state_dict = other.state_dict()
    for name, param in object_.named_parameters():
        if not (torch.allclose(param, torch.zeros_like(param)) or name == "_scale"):
            assert not torch.allclose(param, other_state_dict[name])
    assert count_vars(other) == count_vars(object_)


def _test_from_other_with_copy(object_, class_):
    other = class_.from_other(object_, copy=True)

    assert isinstance(other, class_)
    assert other is not object_
    try:
        other = torch.jit.script(other)
    except:  # noqa: E722
        pass
    other_state_dict = other.state_dict()

    for name, param in object_.named_parameters():
        assert torch.allclose(param, other_state_dict[name])
    assert count_vars(other) == count_vars(object_)


class TestDeterministicNN(object):
    @pytest.fixture(scope="class")
    def net(self):
        return DeterministicNN

    def test_output_shape(
        self, net, in_dim, out_dim, layers, non_linearity, batch_size
    ):
        net = torch.jit.script(
            net(in_dim, out_dim, layers, non_linearity=non_linearity)
        )
        if batch_size is None:
            t = torch.randn(in_dim)
            o = net(t)
            assert o.shape == torch.Size(out_dim)
        else:
            t = torch.randn((batch_size,) + in_dim)
            o = net(t)
            assert o.shape == torch.Size((batch_size,) + out_dim)

    def test_layers(self, net, in_dim, out_dim, layers):
        net = torch.jit.script(net(in_dim, out_dim, layers))
        layers = layers or list()

        # Check nn.parameters (+1: head)
        assert 2 * (len(layers) + 1) == len([*net.parameters()])

        # Check shapes
        layers.append(out_dim[0])
        for i, param in enumerate(net.parameters()):
            assert param.shape[0] == layers[i // 2]

    def test_class_method(self, net, in_dim, out_dim, layers, non_linearity):
        n1 = net(in_dim, out_dim, layers, non_linearity=non_linearity)
        _test_from_other(n1, net)
        _test_from_other_with_copy(n1, net)


class TestHeteroGaussianNN(object):
    @pytest.fixture(scope="class")
    def net(self):
        return HeteroGaussianNN

    def test_output_shape(self, net, in_dim, out_dim, layers, batch_size):
        net = torch.jit.script(net(in_dim, out_dim, layers))
        if batch_size is None:
            t = torch.randn(in_dim)
            o = tensor_to_distribution(net(t)).sample()
            assert o.shape == torch.Size(out_dim)
        else:
            t = torch.randn((batch_size,) + in_dim)
            o = tensor_to_distribution(net(t)).sample()
            assert o.shape == torch.Size((batch_size,) + out_dim)

    def test_output_properties(self, net, in_dim, out_dim, batch_size):
        net = torch.jit.script(net(in_dim, out_dim))
        if batch_size is None:
            t = torch.randn(in_dim)
        else:
            t = torch.randn((batch_size, 2) + in_dim)

        o = tensor_to_distribution(net(t))
        assert isinstance(o, torch.distributions.MultivariateNormal)
        assert o.has_rsample
        assert not o.has_enumerate_support
        assert o.batch_shape == torch.Size(
            (batch_size, 2) if batch_size is not None else []
        )

    def test_layers(self, net, in_dim, out_dim, layers):
        net = torch.jit.script(net(in_dim, out_dim, layers))
        layers = layers or list()

        # Check nn.parameters (+2: mean and covariance)
        assert 2 * (len(layers) + 2) == len([*net.parameters()])

        # Check shapes
        layers.append(out_dim[0])
        layers.append(out_dim[0])
        i = 0
        for name, param in net.named_parameters():
            if name.startswith("_scale"):
                assert param.shape[0] == out_dim[0]  # * out_dim
            else:
                assert param.shape[0] == layers[i // 2]
                i += 1

    def test_class_method(self, net, in_dim, out_dim, layers, non_linearity):
        n1 = net(in_dim, out_dim, layers, non_linearity=non_linearity)
        _test_from_other(n1, net)
        _test_from_other_with_copy(n1, net)


class TestHomoGaussianNN(object):
    @pytest.fixture(scope="class")
    def net(self):
        return HomoGaussianNN

    def test_output_shape(self, net, in_dim, out_dim, layers, batch_size):
        net = torch.jit.script(net(in_dim, out_dim, layers))
        if batch_size is None:
            t = torch.randn(in_dim)
            o = tensor_to_distribution(net(t)).sample()
            assert o.shape == torch.Size(out_dim)
        else:
            t = torch.randn((batch_size,) + in_dim)
            o = tensor_to_distribution(net(t)).sample()
            assert o.shape == torch.Size((batch_size,) + out_dim)

    def test_output_properties(self, net, in_dim, out_dim, batch_size):
        net = torch.jit.script(net(in_dim, out_dim))
        if batch_size is None:
            t = torch.randn(in_dim)
        else:
            t = torch.randn((batch_size, 2) + in_dim)

        o = tensor_to_distribution(net(t))
        assert isinstance(o, torch.distributions.MultivariateNormal)
        assert o.has_rsample
        assert not o.has_enumerate_support
        assert o.batch_shape == torch.Size(
            (batch_size, 2) if batch_size is not None else ()
        )

    def test_layers(self, net, in_dim, out_dim, layers):
        net = torch.jit.script(net(in_dim, out_dim, layers))
        layers = layers or list()

        # Check nn.parameters (+1: mean and covariance has only 1 param)
        assert 2 * (len(layers) + 1) + 1 == len([*net.parameters()])

        # Check shapes
        layers.append(out_dim[0])
        i = 0
        for name, param in net.named_parameters():
            if name.startswith("_scale"):
                assert param.shape[0] == out_dim[0]
            else:
                assert param.shape[0] == layers[i // 2]
                i += 1

    def test_class_method(self, net, in_dim, out_dim, layers, non_linearity):
        n1 = net(in_dim, out_dim, layers, non_linearity=non_linearity)
        _test_from_other(n1, net)
        _test_from_other_with_copy(n1, net)


class TestCategoricalNN(object):
    @pytest.fixture(scope="class")
    def net(self):
        return CategoricalNN

    def test_output_shape(self, net, in_dim, out_dim, layers, batch_size):
        net = torch.jit.script(net(in_dim, out_dim, layers))
        if batch_size is None:
            t = torch.randn(in_dim)
            o = tensor_to_distribution(net(t)).sample()
            assert o.shape == torch.Size([])
        else:
            t = torch.randn((batch_size,) + in_dim)
            o = tensor_to_distribution(net(t)).sample()
            assert o.shape == torch.Size([batch_size])

    def test_output_properties(self, net, in_dim, out_dim, batch_size):
        net = torch.jit.script(net(in_dim, out_dim))
        if batch_size is None:
            t = torch.randn(in_dim)
        else:
            t = torch.randn((batch_size, 2) + in_dim)

        o = tensor_to_distribution(net(t))
        assert isinstance(o, torch.distributions.Categorical)
        assert not o.has_rsample
        assert o.has_enumerate_support
        assert o.batch_shape == torch.Size(
            (batch_size, 2) if batch_size is not None else ()
        )

    def test_layers(self, net, in_dim, out_dim, layers):
        net = torch.jit.script(net(in_dim, out_dim, layers))
        layers = layers or list()

        # Check nn.parameters (+1: head)
        assert 2 * (len(layers) + 1) == len([*net.parameters()])

        # Check shapes
        layers.append(out_dim[0])
        for i, param in enumerate(net.parameters()):
            assert param.shape[0] == layers[i // 2]

    def test_class_method(self, net, in_dim, out_dim, layers, non_linearity):
        n1 = net(in_dim, out_dim, layers, non_linearity=non_linearity)
        _test_from_other(n1, net)
        _test_from_other_with_copy(n1, net)


class TestEnsembleNN(object):
    @pytest.fixture(scope="class", params=[DeterministicNN, HeteroGaussianNN, Ensemble])
    def net(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[True, False])
    def deterministic(self, request):
        return request.param

    def test_num_heads(self, num_heads, deterministic):
        net = Ensemble((4,), (2,), num_heads=num_heads, deterministic=deterministic)
        assert net.num_heads == num_heads

    def test_output_shape(self, out_dim, batch_size, num_heads, deterministic):
        in_dim = (4,)
        net = Ensemble(
            in_dim, out_dim, num_heads=num_heads, deterministic=deterministic
        )
        if batch_size is None:
            t = torch.randn(in_dim)
            o = tensor_to_distribution(net(t)).sample()
            assert o.shape == torch.Size(out_dim)
        else:
            t = torch.randn((batch_size,) + in_dim)
            o = tensor_to_distribution(net(t)).sample()
            assert o.shape == torch.Size((batch_size,) + out_dim)

    def test_output_properties(self, out_dim, num_heads, batch_size, deterministic):
        in_dim = (4,)
        net = Ensemble(
            in_dim, out_dim, num_heads=num_heads, deterministic=deterministic
        )

        if batch_size is None:
            t = torch.randn(in_dim)
        else:
            t = torch.randn((batch_size, 2) + in_dim)

        o = tensor_to_distribution(net(t))
        assert isinstance(o, torch.distributions.MultivariateNormal)
        assert o.has_rsample
        assert not o.has_enumerate_support
        assert o.batch_shape == torch.Size(
            (batch_size, 2) if batch_size is not None else ()
        )

        net.set_prediction_strategy("set_head")
        net.set_head(0)
        o = tensor_to_distribution(net(t))
        if deterministic:
            assert isinstance(o, Delta)
        else:
            assert isinstance(o, torch.distributions.MultivariateNormal)
        assert o.batch_shape == torch.Size(
            (batch_size, 2) if batch_size is not None else ()
        )

        assert o.has_rsample
        assert not o.has_enumerate_support

    def test_layers(self, out_dim, num_heads, layers, deterministic):
        in_dim = (4,)
        net = Ensemble(in_dim, out_dim, layers=layers, num_heads=num_heads)

        layers = layers or list()

        # Check nn.parameters (+1: head)
        assert 2 * (len(layers) + 2) == len([*net.parameters()])

        # Check shapes
        layers.append(out_dim[0] * num_heads)
        layers.append(out_dim[0] * num_heads)

        # Check shapes
        i = 0
        for name, param in net.named_parameters():
            if name.startswith("_scale"):
                assert param.shape[0] == out_dim[0] * num_heads  # * out_dim
            else:
                assert param.shape[0] == layers[i // 2]
                i += 1

    def test_class_method(self, net, batch_size, out_dim, num_heads):
        layers = [64, 64]
        in_dim = (4,)
        try:
            n1 = net(in_dim, out_dim, layers=layers, num_heads=num_heads)
        except TypeError:
            base_net = net(in_dim, out_dim, layers=layers)
            n1 = Ensemble.from_feedforward(base_net, num_heads=num_heads)

            if isinstance(base_net, DeterministicNN):
                assert n1.deterministic
            else:
                assert not n1.deterministic
        _test_from_other(n1, Ensemble)
        _test_from_other_with_copy(n1, Ensemble)

        # Test layers
        layers = layers or list()

        # Check nn.parameters (+1: head)
        assert 2 * (len(layers) + 2) == len([*n1.parameters()])

        # Check shapes
        layers.append(out_dim[0] * num_heads)
        layers.append(out_dim[0] * num_heads)
        i = 0
        for name, param in n1.named_parameters():
            if name.startswith("_scale"):
                assert param.shape[0] == out_dim[0] * num_heads  # * out_dim
            else:
                assert param.shape[0] == layers[i // 2]
                i += 1

        # Check output
        if batch_size is None:
            t = torch.randn(in_dim)
            o = tensor_to_distribution(n1(t))
            assert o.sample().shape == torch.Size(out_dim)
            assert o.batch_shape == torch.Size([])

        else:
            t = torch.randn((batch_size, 2) + in_dim)
            o = tensor_to_distribution(n1(t))
            assert o.sample().shape == torch.Size((batch_size, 2) + out_dim)
            assert o.batch_shape == torch.Size((batch_size, 2))

        assert isinstance(o, torch.distributions.MultivariateNormal)
        assert o.has_rsample
        assert not o.has_enumerate_support


class TestFelixNet(object):
    @pytest.fixture(scope="class")
    def net(self):
        return FelixNet

    def test_output_shape(self, net, in_dim, out_dim, batch_size):
        net = torch.jit.script(net(in_dim, out_dim))
        if batch_size is None:
            t = torch.randn(in_dim)
            o = tensor_to_distribution(net(t)).sample()
            assert o.shape == torch.Size(out_dim)
        else:
            t = torch.randn((batch_size,) + in_dim)
            o = tensor_to_distribution(net(t)).sample()
            assert o.shape == torch.Size((batch_size,) + out_dim)

    def test_output_properties(self, net, in_dim, out_dim, batch_size):
        net = torch.jit.script(net(in_dim, out_dim))
        if batch_size is None:
            t = torch.randn(in_dim)
        else:
            t = torch.randn((batch_size, 2) + in_dim)

        o = tensor_to_distribution(net(t))
        assert isinstance(o, torch.distributions.MultivariateNormal)
        assert o.has_rsample
        assert not o.has_enumerate_support
        assert o.batch_shape == torch.Size(
            (batch_size, 2) if batch_size is not None else ()
        )

    def test_layers(self, net, in_dim, out_dim):
        net = torch.jit.script(net(in_dim, out_dim))
        layers = [64, 64]

        # Check nn.parameters (+2: mean and covariance have only weights)
        assert 2 * (len(layers)) + 2 == len([*net.parameters()])

        # Check shapes
        layers.append(out_dim[0])
        for i, param in enumerate(net.parameters()):
            assert param.shape[0] == layers[i // 2]

    def test_class_method(self, net, in_dim, out_dim):
        n1 = net(in_dim, out_dim)
        _test_from_other(n1, net)
        _test_from_other_with_copy(n1, net)
