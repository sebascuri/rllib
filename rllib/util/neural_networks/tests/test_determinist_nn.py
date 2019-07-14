from rllib.util.neural_networks import DeterministicNN
import torch
import pytest


@pytest.fixture(params=[(32, 2), (64, 4)])
def nn(request):
    in_dim = request.param[0]
    out_dim = request.param[1]

    return DeterministicNN(in_dim=in_dim, out_dim=out_dim), in_dim, out_dim


def test_init(nn):
    pass


def test_forward(nn):
    nn, in_dim, out_dim = nn
    batch_size = 32
    tensor = torch.rand((batch_size, in_dim))

    assert nn.forward(tensor).shape == (batch_size, out_dim)
