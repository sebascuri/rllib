import numpy as np
import pytest
import torch

from rllib.dataset import TrajectoryDataset
from rllib.dataset.datatypes import Observation
from rllib.dataset.transforms import ActionNormalizer, MeanFunction, StateNormalizer


@pytest.fixture(
    params=[(10, 200, 3, 2, 4, 2), (10, 200, 3, 2, 4, 8), (10, 200, 3, 2, 4, None)]
)
def dataset(request):
    num_episodes = request.param[0]
    episode_length = request.param[1]
    state_dim = request.param[2]
    action_dim = request.param[3]
    batch_size = request.param[4]
    sequence_length = request.param[5]

    transforms = [
        MeanFunction(lambda state, action: state),
        StateNormalizer(),
        ActionNormalizer(),
    ]

    dataset = TrajectoryDataset(
        sequence_length=sequence_length, transformations=transforms
    )

    for _ in range(num_episodes):
        trajectory = []
        for i in range(episode_length):
            trajectory.append(
                Observation(
                    state=np.random.randn(state_dim),
                    action=np.random.randn(action_dim),
                    reward=np.random.randn(),
                    next_state=np.random.randn(state_dim),
                    done=i == (episode_length - 1),
                ).to_torch()
            )

        dataset.append(trajectory)
    return (
        dataset,
        num_episodes,
        episode_length,
        state_dim,
        action_dim,
        batch_size,
        sequence_length,
    )


@pytest.fixture(params=[None, 1, 5])
def new_seq_len(request):
    return request.param


def test_length(dataset):
    dataset_ = dataset[0]
    num_episodes = dataset[1]
    episode_length = dataset[2]
    sequence_length = dataset[-1]
    if sequence_length:
        assert len(dataset_) == num_episodes * episode_length // sequence_length
    else:
        assert len(dataset_) == num_episodes


def test_shuffle(dataset):
    dataset = dataset[0]
    dataset.shuffle()


def test_append_error():
    dataset = TrajectoryDataset(sequence_length=10)
    trajectory = [
        Observation(
            np.random.randn(4), np.random.randn(2), 1, np.random.randn(4), True
        ).to_torch()
    ]
    with pytest.raises(ValueError):
        dataset.append(trajectory)


def test_get_item(dataset):
    (
        dataset,
        num_episodes,
        episode_length,
        state_dim,
        action_dim,
        batch_size,
        sequence_length,
    ) = dataset

    for sub_trajectory in dataset:
        assert type(sub_trajectory) is Observation
        batch_len = sequence_length if sequence_length else episode_length
        assert sub_trajectory.state.shape == torch.Size([batch_len, state_dim])
        assert sub_trajectory.action.shape == torch.Size([batch_len, action_dim])
        assert sub_trajectory.next_state.shape == torch.Size([batch_len, state_dim])
        assert sub_trajectory.reward.shape == torch.Size([batch_len])
        assert sub_trajectory.done.shape == torch.Size([batch_len])


def test_sequence_length_setter(dataset, new_seq_len):
    (
        dataset,
        num_episodes,
        episode_length,
        state_dim,
        action_dim,
        batch_size,
        sequence_length,
    ) = dataset
    assert dataset.sequence_length == sequence_length

    dataset.sequence_length = new_seq_len
    assert dataset.sequence_length == new_seq_len
    i = 0
    for sub_trajectory in dataset:
        i += 1
        assert type(sub_trajectory) is Observation
        batch_len = new_seq_len if new_seq_len else episode_length
        assert sub_trajectory.state.shape == torch.Size([batch_len, state_dim])
        assert sub_trajectory.action.shape == torch.Size([batch_len, action_dim])
        assert sub_trajectory.next_state.shape == torch.Size([batch_len, state_dim])
        assert sub_trajectory.reward.shape == torch.Size([batch_len])
        assert sub_trajectory.done.shape == torch.Size([batch_len])

    assert (
        i == (num_episodes * episode_length // new_seq_len)
        if new_seq_len
        else num_episodes
    )


def test_initial_states(dataset):
    (
        dataset,
        num_episodes,
        episode_length,
        state_dim,
        action_dim,
        batch_size,
        sequence_length,
    ) = dataset

    initial_states = dataset.initial_states
    assert initial_states.shape == (num_episodes, state_dim)
