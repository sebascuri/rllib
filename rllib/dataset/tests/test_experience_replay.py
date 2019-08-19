import numpy as np
from rllib.dataset import Observation, ExperienceReplay
from rllib.dataset.transforms import *
import pytest


@pytest.fixture(params=[(10000, 1000), (10000, 100), (100, 101)])
def memory(request):
    max_len = request.param[0]
    number_of_samples = request.param[1]
    state_dim = 3
    action_dim = 2

    transformations = []  # MeanFunction(lambda state, action: state),
    # StateNormalizer(), ActionNormalizer(), RewardClipper(),
    # ]
    memory_ = ExperienceReplay(max_len, transformations)
    for _ in range(number_of_samples):
        memory_.append(Observation(state=np.random.randn(state_dim),
                                   action=np.random.randn(action_dim),
                                   reward=np.random.randn(),
                                   next_state=np.random.randn(state_dim),
                                   done=False)
                       )
    return memory_, max_len, number_of_samples


@pytest.fixture
def experience_replay():
    max_len = 10000
    number_of_samples = 100
    state_dim = 3
    action_dim = 2

    transformations = [MeanFunction(lambda state, action: state),
                       StateNormalizer(), ActionNormalizer(), RewardClipper(),
                       ]
    memory_ = ExperienceReplay(max_len, transformations)
    for _ in range(number_of_samples):
        memory_.append(Observation(state=np.random.randn(state_dim),
                                   action=np.random.randn(action_dim),
                                   reward=np.random.randn(),
                                   next_state=np.random.randn(state_dim),
                                   done=False)
                       )
    return memory_


def test_init(memory):
    pass


def test_len(memory):
    memory, max_len, number_of_samples = memory
    assert len(memory) == np.min((number_of_samples, max_len))


def test_is_full(memory):
    memory, max_len, number_of_samples = memory
    assert memory.is_full == (max_len <= number_of_samples)


def test_get_item(memory):
    memory, max_len, number_of_samples = memory
    for idx in range(len(memory)):
        observation = memory.__getitem__(idx)
        print(observation)
        print(memory._memory[idx])
        print(observation == memory._memory[idx])
        assert observation is memory._memory[idx]
        assert observation == memory._memory[idx]


def test_transforms(experience_replay):
    memory = experience_replay
    for idx in range(len(memory)):
        observation = memory.__getitem__(idx)
        print(observation)
        print(memory._memory[idx])
        print(observation == memory._memory[idx])
        assert observation is not memory._memory[idx]
        assert observation != memory._memory[idx]


def test_shuffle(memory):
    memory, max_len, number_of_samples = memory
    memory.shuffle()
    assert memory.__getitem__(0) is not memory._memory[0]


def test_append_error(experience_replay):
    with pytest.raises(TypeError):
        experience_replay.append((1, 2, 3, 4, 5))
