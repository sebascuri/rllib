import numpy as np
from rllib.dataset import Observation, ExperienceReplay
import pytest


@pytest.fixture(params=[(10000, 1000), (10000, 100), (100, 101)])
def memory(request):
    max_len = request.param[0]
    number_of_samples = request.param[1]
    state_dim = 3
    action_dim = 2
    memory_ = ExperienceReplay(max_len)
    for _ in range(number_of_samples):
        memory_.append(Observation(state=np.random.randn(state_dim),
                                   action=np.random.randn(action_dim),
                                   reward=np.random.randn(),
                                   next_state=np.random.randn(state_dim))
                       )
    return memory_, max_len, number_of_samples


def test_init(memory):
    pass


def test_len(memory):
    memory, max_len, number_of_samples = memory
    assert len(memory) == np.min((number_of_samples, max_len))


def test_is_full(memory):
    memory, max_len, number_of_samples = memory
    assert memory.is_full == (max_len <= number_of_samples)


def test_sample(memory):
    memory, max_len, number_of_samples = memory
    for batch_size in [1, 32]:
        assert memory.sample(batch_size=batch_size).shape == (batch_size,)
