import torch
from rllib.dataset.datatypes import Observation
from rllib.dataset import ExperienceReplay
from rllib.dataset.transforms import *
import pytest
import numpy as np


@pytest.fixture(params=[(10000, 1000), (10000, 100), (100, 101)])
def memory(request):
    max_len = request.param[0]
    number_of_samples = request.param[1]
    state_dim = 3
    action_dim = 2

    transformations = []  # MeanFunction(lambda state, action: state),
    # StateNormalizer(), ActionNormalizer(), RewardClipper(),
    # ]
    memory_ = ExperienceReplay(max_len, transformations=transformations)
    for _ in range(number_of_samples):
        memory_.append(Observation(state=np.random.randn(state_dim),
                                   action=np.random.randn(action_dim),
                                   reward=np.random.randn(),
                                   next_state=np.random.randn(state_dim),
                                   done=False).to_torch()
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
    memory_ = ExperienceReplay(max_len, transformations=transformations)
    for _ in range(number_of_samples):
        memory_.append(Observation(state=np.random.randn(state_dim),
                                   action=np.random.randn(action_dim),
                                   reward=np.random.randn(),
                                   next_state=np.random.randn(state_dim),
                                   done=False).to_torch()
                       )
    return memory_


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
        assert observation is memory.memory[idx]
        assert observation == memory.memory[idx]
        assert type(observation) is Observation
        assert observation.state.shape == torch.Size([3, ])
        assert observation.action.shape == torch.Size([2, ])
        assert observation.next_state.shape == torch.Size([3, ])


def test_iter(memory):
    memory, max_len, number_of_samples = memory
    for idx, observation in enumerate(memory):
        if idx >= len(memory):
            continue
        assert observation is memory.memory[idx]
        assert observation == memory.memory[idx]
        assert observation.state.shape == torch.Size([3, ])
        assert observation.action.shape == torch.Size([2, ])
        assert observation.next_state.shape == torch.Size([3, ])


def test_transforms(experience_replay):
    memory = experience_replay
    for idx in range(len(memory)):
        observation = memory.__getitem__(idx)
        print(observation)
        print(memory.memory[idx])
        assert observation != memory.memory[idx]
        assert observation is not memory.memory[idx]
        assert type(observation) is Observation


def test_append_error(experience_replay):
    with pytest.raises(TypeError):
        experience_replay.append((1, 2, 3, 4, 5))


