from rllib.dataset.utilities import get_backend
from rllib.dataset.transforms.normalizer import Normalizer
import torch
import torch.testing
import pytest


@pytest.fixture(params=[True, False])
def preserve_origin(request):
    return request.param


@pytest.fixture(params=[torch])
def backend(request):
    return request.param


def test_backend(backend):
    assert backend == get_backend(torch.randn(4))


def test_update(backend, preserve_origin):
    array = torch.randn(32, 4)
    transformer = Normalizer(preserve_origin)
    transformer.update(array)
    backend.testing.assert_allclose(transformer._mean, backend.mean(array, 0))
    backend.testing.assert_allclose(transformer._variance, backend.var(array, 0))


def test_normalize(backend, preserve_origin):
    array = torch.randn(32, 4)
    new_array = torch.randn(4)
    transformer = Normalizer(preserve_origin)
    transformer.update(array)
    transformed_array = transformer(new_array)

    if preserve_origin:
        mean = 0
        scale = backend.sqrt(backend.var(array, 0) + backend.mean(array, 0) ** 2)
    else:
        mean = backend.mean(array, 0)
        scale = backend.sqrt(backend.var(array, 0))
    backend.testing.assert_allclose(transformed_array, (new_array - mean) / scale)


def test_unnormalize(backend):
    array = torch.randn(32, 4)
    new_array = torch.randn(4)
    transformer = Normalizer(preserve_origin)
    transformer.update(array)
    transformed_array = transformer(new_array)
    back_array = transformer.inverse(transformed_array)

    backend.testing.assert_allclose(back_array, new_array)


def test_double_update(backend):
    array1 = torch.randn(32, 4)

    transformer = Normalizer()
    transformer.update(array1)

    array2 = torch.randn(16, 4)
    transformer.update(array2)

    array = backend.stack((*array1, *array2))

    backend.testing.assert_allclose(transformer._mean, backend.mean(array, 0))
    backend.testing.assert_allclose(transformer._variance, backend.var(array, 0))
