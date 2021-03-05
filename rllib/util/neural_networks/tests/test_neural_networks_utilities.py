import pytest
import torch
import torch.functional
import torch.nn as nn
import torch.testing

from rllib.util.neural_networks import (
    CategoricalNN,
    DeterministicNN,
    Ensemble,
    FelixNet,
    HeteroGaussianNN,
    HomoGaussianNN,
)
from rllib.util.neural_networks.utilities import (
    TileCode,
    get_batch_size,
    init_head_bias,
    init_head_weight,
    inverse_softplus,
    one_hot_encode,
    random_tensor,
    update_parameters,
    zero_bias,
)


def test_inverse_softplus():
    t = torch.randn(32, 4)
    torch.testing.assert_allclose(t, inverse_softplus(nn.functional.softplus(t)))


def test_zero_bias():
    in_dim = (4,)
    out_dim = (2,)
    layers = [2, 4, 6, 8, 4, 2]
    n = DeterministicNN(in_dim, out_dim, layers)
    zero_bias(n)

    for name, param in n.named_parameters():
        if name.endswith("bias"):
            torch.testing.assert_allclose(param, torch.zeros_like(param.data))


def test_init_head():
    in_dim = (4,)
    out_dim = (2,)
    layers = [2, 4, 6, 8, 4, 2]
    net = DeterministicNN(in_dim, out_dim, layers)
    init_head_bias(net, offset=1, delta=0)
    init_head_weight(net)

    for name, param in net.named_parameters():
        if name.endswith("head.bias"):
            torch.testing.assert_allclose(param, torch.ones_like(param.data))


class TestUpdateParams(object):
    @pytest.fixture(params=[1.0, 0.9, 0.5, 0.2, 0.0], scope="class")
    def tau(self, request):
        return request.param

    @pytest.fixture(
        params=[
            DeterministicNN,
            FelixNet,
            CategoricalNN,
            Ensemble,
            HomoGaussianNN,
            HeteroGaussianNN,
        ],
        scope="class",
    )
    def network(self, request):
        return request.param

    def test_network(self, tau, network):
        class_ = network
        in_dim = (16,)
        out_dim = (4,)
        layers = [32, 4]
        if class_ is Ensemble:
            net1 = class_(in_dim, out_dim, num_heads=5, layers=layers)
            net1c = class_(in_dim, out_dim, num_heads=5, layers=layers)
            net2 = class_(in_dim, out_dim, num_heads=5, layers=layers)
            net2c = class_(in_dim, out_dim, num_heads=5, layers=layers)
        elif class_ is FelixNet:
            net1 = class_(in_dim, out_dim)
            net1c = class_(in_dim, out_dim)
            net2 = class_(in_dim, out_dim)
            net2c = class_(in_dim, out_dim)
        else:
            net1 = class_(in_dim, out_dim, layers)
            net1c = class_(in_dim, out_dim, layers)
            net2 = class_(in_dim, out_dim, layers)
            net2c = class_(in_dim, out_dim, layers)

        net1c.load_state_dict(net1.state_dict())
        net2c.load_state_dict(net2.state_dict())

        update_parameters(net1, net2, tau)

        for name, _ in net1.named_parameters():
            param1 = net1.state_dict()[name]
            param1c = net1c.state_dict()[name]

            param2 = net2.state_dict()[name]
            param2c = net2c.state_dict()[name]

            torch.testing.assert_allclose(
                param1.data, tau * param1c.data + (1 - tau) * param2c.data
            )

            torch.testing.assert_allclose(param2.data, param2c.data)

            assert param1 is not param1c
            if tau == 0:
                assert torch.allclose(param1.data, param2c.data)
            elif tau == 1:
                assert torch.allclose(param1.data, param1c.data)
            elif not (
                torch.allclose(param1.data, torch.zeros_like(param1.data))
                or name == "_scale"
            ):
                assert not (torch.allclose(param1.data, param1c.data))
                assert not (torch.allclose(param2.data, param1c.data))


class TestTileCode(object):
    @pytest.fixture(params=[True, False], scope="class")
    def one_hot(self, request):
        return request.param

    def test_1d(self, one_hot):
        encoding = TileCode([-1.0], [1.0], bins=10, one_hot=one_hot)
        encoding = torch.jit.script(encoding)
        t_in = torch.linspace(-1.0, 1.0, 11).unsqueeze(-1)
        code = encoding(t_in)

        if one_hot:
            assert encoding.extra_dims == 10
            torch.testing.assert_allclose(code, torch.eye(11))
        else:
            assert encoding.extra_dims == 0
            torch.testing.assert_allclose(code, torch.arange(11).long())

        t_in_eps = t_in + 0.2 * (torch.rand_like(t_in) - 0.5)
        code_eps = encoding(t_in_eps)

        torch.testing.assert_allclose(code, code_eps)

        torch.testing.assert_allclose(
            encoding.tiles[..., 0], torch.linspace(-0.9, 0.9, 10)
        )

    def test_2d(self, one_hot):
        encoding = TileCode([-1.0, -2.0], [1.0, 2.0], bins=10, one_hot=one_hot)
        encoding = torch.jit.script(encoding)

        t_in = torch.cartesian_prod(
            torch.linspace(-1.0, 1.0, 11), torch.linspace(-2.0, 2.0, 11)
        )
        code = encoding(t_in)

        if one_hot:
            assert encoding.extra_dims == 11 * 11 - 2
            torch.testing.assert_allclose(code, torch.eye(11 * 11))
        else:
            assert encoding.extra_dims == -1
            torch.testing.assert_allclose(code, torch.arange(11 * 11).long())

        t_in_eps = t_in + (torch.rand_like(t_in) - 0.5) @ torch.diag(
            torch.tensor([0.2, 0.4])
        )
        code_eps = encoding(t_in_eps)

        torch.testing.assert_allclose(code, code_eps)

        torch.testing.assert_allclose(
            encoding.tiles[..., 0], torch.linspace(-0.9, 0.9, 10)
        )
        torch.testing.assert_allclose(
            encoding.tiles[..., 1], torch.linspace(-1.8, 1.8, 10)
        )

    def test_3d(self, one_hot):
        encoding = TileCode(
            [-1.0, -5.0, -3.0], [1.0, 5.0, 5.0], bins=5, one_hot=one_hot
        )
        encoding = torch.jit.script(encoding)

        t_in = torch.cartesian_prod(
            torch.linspace(-1.0, 1.0, 6),
            torch.linspace(-5.0, 5.0, 6),
            torch.linspace(-3.0, 5.0, 6),
        )
        code = encoding(t_in)

        if one_hot:
            assert encoding.extra_dims == 6 ** 3 - 3
            torch.testing.assert_allclose(code, torch.eye(6 ** 3))
        else:
            assert encoding.extra_dims == -2
            torch.testing.assert_allclose(code, torch.arange(6 ** 3))

        t_in_eps = t_in + (torch.rand_like(t_in) - 0.5) @ torch.diag(
            torch.tensor([0.2, 1.0, 0.8])
        )
        code_eps = encoding(t_in_eps)

        torch.testing.assert_allclose(code, code_eps)

        torch.testing.assert_allclose(
            encoding.tiles[..., 0], torch.linspace(-0.8, 0.8, 5)
        )
        torch.testing.assert_allclose(encoding.tiles[..., 1], torch.linspace(-4, 4, 5))
        torch.testing.assert_allclose(
            encoding.tiles[..., 2], torch.linspace(-2.2, 4.2, 5)
        )


class TestOneHotEncode(object):
    @pytest.fixture(params=[None, 1, 16], scope="class")
    def batch_size(self, request):
        return request.param

    def test_output_dimension(self, batch_size):
        num_classes = 4
        tensor = torch.randint(4, torch.Size([batch_size] if batch_size else []))
        out_tensor = one_hot_encode(tensor, num_classes)
        assert out_tensor.dim() == 2 if batch_size else 1

    def test_double_batch(self):
        num_classes = 4
        tensor = torch.randint(num_classes, torch.Size([32, 5]))
        out_tensor = one_hot_encode(tensor, num_classes)
        assert out_tensor.dim() == 3
        assert out_tensor.shape == torch.Size([32, 5, 4])

    def test_correctness(self):
        tensor = torch.tensor([2])
        out_tensor = one_hot_encode(tensor, 4)
        torch.testing.assert_allclose(out_tensor, torch.tensor([[0.0, 0.0, 1.0, 0.0]]))

        tensor = torch.tensor([2, 1])
        out_tensor = one_hot_encode(tensor, 4)
        torch.testing.assert_allclose(
            out_tensor, torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        )

    def test_output_sum_one(self, batch_size):
        num_classes = 4
        tensor = torch.randint(4, torch.Size([batch_size] if batch_size else []))
        out_tensor = one_hot_encode(tensor, num_classes)
        if batch_size:
            torch.testing.assert_allclose(
                out_tensor.sum(dim=-1), torch.ones(batch_size)
            )
        else:
            torch.testing.assert_allclose(out_tensor.sum(dim=-1), torch.tensor(1.0))

    def test_indexing(self, batch_size):
        num_classes = 4
        tensor = torch.randint(4, torch.Size([batch_size] if batch_size else [1]))
        out_tensor = one_hot_encode(tensor, num_classes)
        indexes = out_tensor.gather(-1, tensor.unsqueeze(-1)).long().squeeze(-1)
        torch.testing.assert_allclose(indexes, torch.ones_like(tensor))


class TestGetBatchSize(object):
    @pytest.fixture(params=[None, 1, 4, 16], scope="class")
    def batch_size(self, request):
        return request.param

    def test_discrete(self, batch_size):
        size = (batch_size,) if batch_size else ()
        tensor = torch.randint(4, size)
        if batch_size:
            assert get_batch_size(tensor, ()) == (batch_size,)
        else:
            assert get_batch_size(tensor, ()) == ()

    def test_continuous(self, batch_size):
        if batch_size:
            tensor = torch.randn(batch_size, 4)
            assert get_batch_size(tensor, (4,)) == (batch_size,)
        else:
            tensor = torch.randn(4)
            assert get_batch_size(tensor, (4,)) == ()

    def test_images(self, batch_size):
        if batch_size:
            tensor = torch.randn(batch_size, 32, 32, 3)
            assert get_batch_size(tensor, (32, 32, 3)) == (batch_size,)
        else:
            tensor = torch.randn(32, 32, 3)
            assert get_batch_size(tensor, (32, 32, 3)) == ()


class TestRandomTensor(object):
    @pytest.fixture(params=[None, 1, 4, 16], scope="class")
    def batch_size(self, request):
        return request.param

    @pytest.fixture(params=[2, 4], scope="class")
    def dim(self, request):
        return request.param

    def test_discrete(self, batch_size, dim):
        tensor = random_tensor(True, dim, batch_size)
        assert tensor.dtype is torch.long
        if batch_size:
            assert tensor.dim() == 1
            assert tensor.shape == (batch_size,)
        else:
            assert tensor.dim() == 0
            assert tensor.shape == ()

    def test_continuous(self, batch_size, dim):
        tensor = random_tensor(False, dim, batch_size)
        assert tensor.dtype is torch.get_default_dtype()
        if batch_size:
            assert tensor.dim() == 2
            assert tensor.shape == (batch_size, dim)
        else:
            assert tensor.dim() == 1
            assert tensor.shape == (dim,)
