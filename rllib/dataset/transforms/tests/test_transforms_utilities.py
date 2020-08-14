import pytest
import torch
import torch.testing

from rllib.dataset.transforms.normalizer import Normalizer
from rllib.util.utilities import get_backend


@pytest.fixture(params=[True, False])
def preserve_origin(request):
    return request.param


def test_backend():
    assert torch == get_backend(torch.randn(4))


def test_update(preserve_origin):
    array = torch.randn(32, 4)
    transformer = Normalizer(preserve_origin)
    transformer.update(array)
    torch.testing.assert_allclose(transformer.mean, torch.mean(array, 0))
    torch.testing.assert_allclose(transformer.variance, torch.var(array, 0))


def test_normalize(preserve_origin):
    array = torch.randn(32, 4)
    new_array = torch.randn(4)
    transformer = Normalizer(preserve_origin)
    transformer.update(array)
    transformed_array = transformer(new_array)

    if preserve_origin:
        mean = 0
        scale = torch.sqrt(torch.var(array, 0) + torch.mean(array, 0) ** 2)
    else:
        mean = torch.mean(array, 0)
        scale = torch.sqrt(torch.var(array, 0))
    torch.testing.assert_allclose(transformed_array, (new_array - mean) / scale)


def test_unnormalize():
    array = torch.randn(32, 4)
    new_array = torch.randn(4)
    transformer = Normalizer(preserve_origin)
    transformer.update(array)
    transformed_array = transformer(new_array)
    back_array = transformer.inverse(transformed_array)

    torch.testing.assert_allclose(back_array, new_array)


def test_sequential_update():
    torch.manual_seed(0)
    array = torch.randn(16, 4)

    transformer = Normalizer()
    transformer.update(array)

    torch.testing.assert_allclose(transformer.mean, torch.mean(array, 0))
    torch.testing.assert_allclose(transformer.variance, torch.var(array, 0))

    for _ in range(10):
        new_array = torch.randn(torch.randint(1, 32, (1,)), 4)
        transformer.update(new_array)

        array = torch.cat((array, new_array), dim=0)
        torch.testing.assert_allclose(transformer.mean, torch.mean(array, 0))
        torch.testing.assert_allclose(transformer.variance, torch.var(array, 0))
