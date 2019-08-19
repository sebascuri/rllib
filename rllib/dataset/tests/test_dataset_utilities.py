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


def test_sack_list_of_tuples(trajectory):
    for backend in [np, torch]:
        stacked_trajectory = stack_list_of_tuples(trajectory, backend=backend)
        assert type(stacked_trajectory) is Observation
        assert stacked_trajectory.state.shape == (3, 4)
        assert stacked_trajectory.action.shape == (3, 2)
        assert stacked_trajectory.next_state.shape == (3, 4)
        assert stacked_trajectory.reward.shape == (3, )
        assert stacked_trajectory.done.shape == (3, )
