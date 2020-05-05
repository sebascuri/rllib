import numpy as np
import pytest
import torch

from rllib.dataset import ExperienceReplay
from rllib.dataset.datatypes import RawObservation, Observation
from rllib.dataset.transforms import MeanFunction, RewardClipper, StateNormalizer, \
    ActionNormalizer


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
        memory_.append(RawObservation(state=np.random.randn(state_dim),
                                      action=np.random.randn(action_dim),
                                      reward=np.random.randn(),
                                      next_state=np.random.randn(state_dim),
                                      done=False).to_torch()
                       )
    return memory_, max_len, number_of_samples


@pytest.fixture(params=[1, 3])
def n_steps(request):
    return request.param


def experience_replay(n_steps):
    max_len = 10000
    number_of_samples = 100
    state_dim = 3
    action_dim = 2

    transformations = [MeanFunction(lambda state, action: state),
                       StateNormalizer(), ActionNormalizer(), RewardClipper(),
                       ]
    memory_ = ExperienceReplay(max_len, transformations=transformations,
                               num_steps=n_steps)
    for _ in range(number_of_samples):
        memory_.append(RawObservation(state=np.random.randn(state_dim),
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
        observation, idx, weight = memory.__getitem__(idx)
        assert idx == idx
        assert weight == 1.
        assert observation is memory.memory[idx]
        assert observation == memory.memory[idx]
        assert type(observation) is Observation
        assert observation.state.shape == torch.Size([1, 3])
        assert observation.action.shape == torch.Size([1, 2])
        assert observation.next_state.shape == torch.Size([1, 3])


def test_n_steps(n_steps):
    er = experience_replay(n_steps)

    for idx in range(len(er)):
        observation, idx, weight = er.__getitem__(idx)
        assert idx == idx
        assert weight == 1.
        assert type(observation) is Observation
        assert observation is not er.memory[idx]
        assert type(observation) is Observation

        if idx < len(er) - 1:
            assert observation.state.shape == torch.Size([n_steps, 3])
            assert observation.action.shape == torch.Size([n_steps, 2])
            assert observation.next_state.shape == torch.Size([n_steps, 3])
        else:
            assert observation.state.shape == torch.Size(
                [len(er) % n_steps + n_steps, 3])
            assert observation.action.shape == torch.Size(
                [len(er) % n_steps + n_steps, 2])
            assert observation.next_state.shape == torch.Size(
                [len(er) % n_steps + n_steps, 3])


def test_iter(memory):
    memory, max_len, number_of_samples = memory
    for idx, (observation, idx_, weight) in enumerate(memory):
        if idx >= len(memory):
            continue
        assert idx == idx_
        assert weight == 1.
        assert observation is memory.memory[idx]
        assert observation == memory.memory[idx]
        assert observation.state.shape == torch.Size([1, 3, ])
        assert observation.action.shape == torch.Size([1, 2, ])
        assert observation.next_state.shape == torch.Size([1, 3, ])


def test_append_error():
    with pytest.raises(TypeError):
        experience_replay(1).append((1, 2, 3, 4, 5))
