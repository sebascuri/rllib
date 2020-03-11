from rllib.dataset.utilities import get_backend
from rllib.dataset.transforms.normalizer import Normalizer
import torch
import torch.testing
import numpy as np
import pytest


@pytest.fixture(params=[True, False])
def preserve_origin(request):
    return request.param


@pytest.fixture(params=[torch, np])
def backend(request):
    return request.param


def get_rand_module(backend):
    if backend == torch:
        rand = torch.randn
    else:
        rand = np.random.randn
    return rand


def test_backend(backend):
    rand = get_rand_module(backend)
    assert backend == get_backend(rand(4))


def test_update(backend, preserve_origin):
    rand = get_rand_module(backend)
    array = rand(32, 4)
    transformer = Normalizer(preserve_origin)
    transformer.update(array)
    backend.testing.assert_allclose(transformer._mean, backend.mean(array, 0))
    backend.testing.assert_allclose(transformer._variance, backend.var(array, 0))


def test_normalize(backend, preserve_origin):
    rand = get_rand_module(backend)
    array = rand(32, 4)
    new_array = rand(4)
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
    rand = get_rand_module(backend)
    array = rand(32, 4)
    new_array = rand(4)
    transformer = Normalizer(preserve_origin)
    transformer.update(array)
    transformed_array = transformer(new_array)
    back_array = transformer.inverse(transformed_array)

    backend.testing.assert_allclose(back_array, new_array)


def test_double_update(backend):
    rand = get_rand_module(backend)
    array1 = rand(32, 4)

    transformer = Normalizer()
    transformer.update(array1)

    array2 = rand(16, 4)
    transformer.update(array2)

    array = backend.stack((*array1, *array2))

    backend.testing.assert_allclose(transformer._mean, backend.mean(array, 0))
    backend.testing.assert_allclose(transformer._variance, backend.var(array, 0))
