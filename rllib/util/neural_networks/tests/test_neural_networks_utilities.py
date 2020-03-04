import torch
import torch.testing
import torch.functional
import torch.nn as nn
import pytest
from rllib.util.neural_networks.utilities import *
from rllib.util.neural_networks import DeterministicNN


def test_inverse_softplus():
    t = torch.randn(32, 4)
    torch.testing.assert_allclose(t, inverse_softplus(nn.functional.softplus(t)))


def test_zero_bias():
    in_dim = 4
    out_dim = 2
    layers = [2, 4, 6, 8, 4, 2]
    n = DeterministicNN(in_dim, out_dim, layers)
    zero_bias(n)

    for name, param in n.named_parameters():
        if name.endswith('bias'):
            torch.testing.assert_allclose(param, torch.zeros_like(param.data))


class TestUpdateParams(object):
    @pytest.fixture(params=[1., 0.9, 0.5, 0.2], scope="class")
    def tau(self, request):
        return request.param

    def test_network(self, tau):
        in_dim = 16
        out_dim = 4
        layers = [32, 4]
        net1 = DeterministicNN(in_dim, out_dim, layers)
        net1c = DeterministicNN(in_dim, out_dim, layers)
        net1c.load_state_dict(net1.state_dict())

        net2 = DeterministicNN(in_dim, out_dim, layers)
        net2c = DeterministicNN(in_dim, out_dim, layers)
        net2c.load_state_dict(net2.state_dict())

        update_parameters(net1.parameters(), net2.parameters(), tau)

        for param1, param1c, param2, param2c in zip(net1.parameters(),
                                                    net1c.parameters(),
                                                    net2.parameters(),
                                                    net2c.parameters()
                                                    ):
            torch.testing.assert_allclose(
                param1.data, tau * param2c.data + (1-tau) * param1c.data
            )

            torch.testing.assert_allclose(
                param2.data, param2c.data
            )
            if tau < 1:
                assert not (torch.allclose(param1.data, param1c.data))

    def test_parameter(self, tau):
        t1_data = [32, 16, 0.3]
        t2_data = [0., 0., 0.5]
        t1 = nn.Parameter(torch.tensor(t1_data))
        t2 = nn.Parameter(torch.tensor(t2_data))

        update_parameters([t1], [t2], tau)
        torch.testing.assert_allclose(t1, nn.Parameter(
            tau * torch.tensor(t2_data) + (1-tau) * torch.tensor(t1_data)))

        torch.testing.assert_allclose(t2, nn.Parameter(torch.tensor(t2_data)))

        update_parameters([t1], [t1], tau)
        torch.testing.assert_allclose(t1, nn.Parameter(
            tau * torch.tensor(t2_data) + (1 - tau) * torch.tensor(t1_data)))

        torch.testing.assert_allclose(t2, nn.Parameter(torch.tensor(t2_data)))


class TestOneHoteEncode(object):
    @pytest.fixture(params=[None, 1, 16], scope="class")
    def batch_size(self, request):
        return request.param

    def test_output_dimension(self, batch_size):
        num_classes = 4
        tensor = torch.randint(4, torch.Size([batch_size] if batch_size else []))
        out_tensor = one_hot_encode(tensor, num_classes)
        assert out_tensor.dim() == 2 if batch_size else 1

        tensor = torch.randint(4, torch.Size([batch_size, 1] if batch_size else [1]))
        out_tensor = one_hot_encode(tensor, num_classes)
        assert out_tensor.dim() == 2 if batch_size else 1

    def test_error(self, batch_size):
        with pytest.raises(TypeError):
            one_hot_encode(torch.randn(4), 4)

    def test_correctness(self):
        tensor = torch.tensor([2])
        out_tensor = one_hot_encode(tensor, 4)
        torch.testing.assert_allclose(out_tensor, torch.tensor([0., 0., 1., 0.]))

        tensor = torch.tensor([2, 1])
        out_tensor = one_hot_encode(tensor, 4)
        torch.testing.assert_allclose(out_tensor, torch.tensor([[0., 0., 1., 0.],
                                                                [0., 1., 0., 0.]]))

    def test_output_sum_one(self, batch_size):
            num_classes = 4
            tensor = torch.randint(4, torch.Size([batch_size] if batch_size else []))
            out_tensor = one_hot_encode(tensor, num_classes)
            torch.testing.assert_allclose(
                out_tensor.sum(dim=-1),
                torch.ones(batch_size if batch_size else 1).squeeze(-1)
            )

    def test_indexing(self, batch_size):
        num_classes = 4
        tensor = torch.randint(4, torch.Size([batch_size, 1] if batch_size else [1]))
        out_tensor = one_hot_encode(tensor, num_classes)
        if batch_size:
            indexes = out_tensor.gather(-1, tensor).long()
        else:
            indexes = out_tensor.gather(-1, tensor.unsqueeze(-1)).long()
        torch.testing.assert_allclose(indexes, torch.ones_like(tensor))


class TestGetBatchSize(object):
    @pytest.fixture(params=[None, 1, 4, 16], scope="class")
    def batch_size(self, request):
        return request.param

    def test_discrete(self, batch_size):
        size = (batch_size,) if batch_size else ()
        tensor = torch.randint(4, size)

        assert batch_size == get_batch_size(tensor, is_discrete=True)
        assert batch_size == get_batch_size(tensor)

    def test_continuous(self, batch_size):
        if batch_size:
            tensor = torch.randn(batch_size, 4)
        else:
            tensor = torch.randn(4)

        assert batch_size == get_batch_size(tensor, is_discrete=False)
        assert batch_size == get_batch_size(tensor)


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
            assert tensor.shape == (batch_size, )
        else:
            assert tensor.dim() == 0
            assert tensor.shape == ()

    def test_continuous(self, batch_size, dim):
        tensor = random_tensor(False, dim, batch_size)
        assert tensor.dtype is torch.float
        if batch_size:
            assert tensor.dim() == 2
            assert tensor.shape == (batch_size, dim,)
        else:
            assert tensor.dim() == 1
            assert tensor.shape == (dim,)
