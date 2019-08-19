import numpy as np
import torch
from rllib.dataset import Observation, TrajectoryDataset
from rllib.dataset.transforms import *
import pytest


@pytest.fixture(params=[(10, 200, 3, 2, 4, 2), (10, 200, 3, 2, 4, 8)])
def dataset(request):
    num_episodes = request.param[0]
    episode_length = request.param[1]
    state_dim = request.param[2]
    action_dim = request.param[3]
    batch_size = request.param[4]
    sequence_length = request.param[5]

    transforms = [MeanFunction(lambda state, action: state),
                  StateNormalizer(), ActionNormalizer()]

    dataset = TrajectoryDataset(sequence_length=sequence_length,
                                transformations=transforms)

    for _ in range(num_episodes):
        trajectory = []
        for i in range(episode_length):
            trajectory.append(
                Observation(state=np.random.randn(state_dim),
                            action=np.random.randn(action_dim),
                            reward=np.random.randn(),
                            next_state=np.random.randn(state_dim),
                            done=i == (episode_length - 1))
            )

        dataset.append(trajectory)
    return (dataset, num_episodes, episode_length, state_dim, action_dim, batch_size,
            sequence_length)


def test_length(dataset):
    dataset_ = dataset[0]
    num_episodes = dataset[1]
    episode_length = dataset[2]
    sequence_length = dataset[-1]
    assert len(dataset_) == num_episodes * episode_length // sequence_length


def test_shuffle(dataset):
    dataset = dataset[0]
    dataset.shuffle()


def test_get_item(dataset):
    (dataset, num_episodes, episode_length, state_dim, action_dim, batch_size,
     sequence_length) = dataset
    for sub_trajectory in dataset:
        assert type(sub_trajectory) is Observation
        assert sub_trajectory.state.shape == torch.Size([sequence_length, state_dim])
        assert sub_trajectory.action.shape == torch.Size([sequence_length, action_dim])
        assert sub_trajectory.next_state.shape == torch.Size([sequence_length, state_dim])
        assert sub_trajectory.reward.shape == torch.Size([sequence_length, ])
        assert sub_trajectory.done.shape == torch.Size([sequence_length, ])
