import numpy as np
import torch

from rllib.dataset.datatypes import Observation
from rllib.dataset.utilities import stack_list_of_tuples


def get_trajectory():
    t = []
    for reward in [3.0, -2.0, 0.5]:
        t.append(
            Observation(
                state=torch.randn(4),
                action=torch.randn(2),
                reward=reward,
                next_state=torch.randn(4),
                done=False,
            )
        )
    return t


def test_stack_list_of_observations():
    trajectory = get_trajectory()
    stacked_trajectory = stack_list_of_tuples(trajectory)
    stacked_trajectory = stacked_trajectory.to_torch()
    assert type(stacked_trajectory) is Observation
    assert stacked_trajectory.state.shape == (3, 4)
    assert stacked_trajectory.action.shape == (3, 2)
    assert stacked_trajectory.next_state.shape == (3, 4)
    assert stacked_trajectory.reward.shape == (3,)
    assert stacked_trajectory.done.shape == (3,)
    for val in stacked_trajectory:
        assert val.dtype is torch.get_default_dtype()


def test_stack_list_of_lists():
    trajectory = [[1, 2, 3, 4], [20, 30, 40, 50], [3, 4, 5, 6], [40, 50, 60, 70]]
    stacked_trajectory = stack_list_of_tuples(trajectory)

    np.testing.assert_allclose(stacked_trajectory[0], np.array([1, 20, 3, 40]))
    np.testing.assert_allclose(stacked_trajectory[1], np.array([2, 30, 4, 50]))
    np.testing.assert_allclose(stacked_trajectory[2], np.array([3, 40, 5, 60]))
    np.testing.assert_allclose(stacked_trajectory[3], np.array([4, 50, 6, 70]))
