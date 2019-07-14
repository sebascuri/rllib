from rllib.util.neural_networks import GaussianNN, CategoricalNN
import torch
import pytest


@pytest.fixture(params=[(32, 2), (64, 4)])
def gaussian_nn(request):
    in_dim = request.param[0]
    out_dim = request.param[1]

    return GaussianNN(in_dim=in_dim, out_dim=out_dim), in_dim, out_dim


@pytest.fixture(params=[(32, 2), (64, 4)])
def categorical_nn(request):
    in_dim = request.param[0]
    out_dim = request.param[1]

    return CategoricalNN(in_dim=in_dim, out_dim=out_dim), in_dim, out_dim


def test_init(gaussian_nn, categorical_nn):
    pass


def test_gaussian_forward(gaussian_nn):
    nn, in_dim, out_dim = gaussian_nn
    batch_size = 32
    tensor = torch.rand((batch_size, in_dim))
    distribution = nn.forward(tensor)

    assert distribution.mean.shape == (batch_size, out_dim)
    assert distribution.has_rsample
    assert not distribution.has_enumerate_support


def test_categorical_forward(categorical_nn):
    nn, in_dim, out_dim = categorical_nn
    batch_size = 32
    tensor = torch.rand((batch_size, in_dim))
    distribution = nn.forward(tensor)

    assert distribution.logits.shape == (batch_size, out_dim)
    assert not distribution.has_rsample
    assert distribution.has_enumerate_support
