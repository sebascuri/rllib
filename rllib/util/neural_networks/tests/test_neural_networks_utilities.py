import torch
import torch.testing
import torch.functional
import torch.nn as nn
import pytest
from rllib.util.neural_networks.utilities import inverse_softplus, zero_bias, update_parameters
from rllib.util.neural_networks import DeterministicNN


def test_inverse_softplus():
    t = torch.randn(32, 4)
    torch.testing.assert_allclose(t, inverse_softplus(nn.functional.softplus(t)))


def test_zero_bias():
    in_dim = 4
    out_dim = 2
    layers = [2, 4, 6, 8, 4, 2]
    n = DeterministicNN(in_dim, out_dim, layers)
    zero_bias(n.named_parameters())

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




