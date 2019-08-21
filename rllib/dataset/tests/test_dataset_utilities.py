import numpy as np
import torch
import pytest
from rllib.dataset import Observation
from rllib.dataset import stack_list_of_tuples


def _trajectory(backend):
    t = []
    for reward in [3., -2., 0.5]:
        t.append(get_observation(backend, reward))
    return t


def get_observation(backend, reward=None):
    if backend is np:
        rand = np.random.randn
    else:
        rand = torch.randn

    return Observation(state=rand(4),
                       action=rand(2),
                       reward=reward if reward else rand(1),
                       next_state=rand(4),
                       done=False)


@pytest.fixture(params=[np, torch])
def trajectory(request):
    return _trajectory(request.param)


def test_sack_list_of_observations(trajectory):
    for backend in [np, torch]:
        stacked_trajectory = stack_list_of_tuples(trajectory, backend=backend)
        assert type(stacked_trajectory) is Observation
        assert stacked_trajectory.state.shape == (3, 4)
        assert stacked_trajectory.action.shape == (3, 2)
        assert stacked_trajectory.next_state.shape == (3, 4)
        assert stacked_trajectory.reward.shape == (3, )
        assert stacked_trajectory.done.shape == (3, )


def test_sack_list_of_lists():
    trajectory = [[1, 2, 3, 4], [20, 30, 40, 50], [3, 4, 5, 6], [40, 50, 60, 70]]
    for backend in [np, torch]:
        stacked_trajectory = stack_list_of_tuples(trajectory, backend=backend)

        np.testing.assert_allclose(stacked_trajectory[0], np.array([1, 20, 3, 40]))
        np.testing.assert_allclose(stacked_trajectory[1], np.array([2, 30, 4, 50]))
        np.testing.assert_allclose(stacked_trajectory[2], np.array([3, 40, 5, 60]))
        np.testing.assert_allclose(stacked_trajectory[3], np.array([4, 50, 6, 70]))
