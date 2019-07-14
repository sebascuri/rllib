import numpy as np
from rllib.dataset import Observation, Dataset
import pytest


@pytest.fixture(params=[(10, 200, 3, 2, 4, 2), (10, 200, 3, 2, 4, 8)])
def dataset(request):
    num_episodes = request.param[0]
    episode_length = request.param[1]
    state_dim = request.param[2]
    action_dim = request.param[3]
    batch_size = request.param[4]
    sequence_length = request.param[5]

    dataset = Dataset(state_dim=state_dim, action_dim=action_dim,
                      batch_size=batch_size, sequence_length=sequence_length)

    for _ in range(num_episodes):
        trajectory = []
        for _ in range(episode_length):
            trajectory.append(
                Observation(state=np.random.randn(state_dim),
                            action=np.random.randn(action_dim),
                            reward=np.random.randn(),
                            next_state=np.random.randn(state_dim))
            )

        dataset.append(trajectory)
    return (dataset, num_episodes, episode_length, state_dim, action_dim, batch_size,
            sequence_length)


def test_properties(dataset):
    num_episodes = dataset[1]
    episode_length = dataset[2]
    dataset = dataset[0]
    sequence_length = (num_episodes * episode_length) // dataset.number_sub_trajectories
    assert dataset.number_sub_trajectories == (num_episodes
                                               * (episode_length // sequence_length))
    assert dataset.number_trajectories == num_episodes


def test_length(dataset):
    dataset = dataset[0]
    assert len(dataset) == dataset.number_sub_trajectories


def test_shuffle(dataset):
    dataset = dataset[0]
    dataset.shuffle()


def test_split(dataset):
    dataset = dataset[0]
    split_ratio = 0.8
    train, test = dataset.split(split_ratio, shuffle=True)

    assert len(train) == int(split_ratio * len(dataset))
    assert len(test) == int((1 - split_ratio) * len(dataset)) or len(test) == 1 + int(
        (1 - split_ratio) * len(dataset))


def test_iter(dataset):
    state_dim = dataset[3]
    action_dim = dataset[4]
    batch_size = dataset[5]
    sequence_length = dataset[6]
    dataset = dataset[0]

    batches = 0
    for state, action, reward, next_state in dataset:
        batches += 1
        assert state.shape == (batch_size, sequence_length, state_dim)
        assert action.shape == (batch_size, sequence_length, action_dim)
        assert reward.shape == (batch_size, sequence_length, 1)
        assert next_state.shape == (batch_size, sequence_length, state_dim)

    assert batches == len(dataset) // batch_size
