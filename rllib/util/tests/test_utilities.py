# if __name__ == "__main__":
#     from rllib.dataset import Observation
#     trajectory = []
#     for i in range(10):
#         observation = Observation(state=0, action=1, next_state=2, reward=1, done=0)
#         trajectory.append(observation)
#
#     gamma = 0.9
#     print(mc_value(trajectory, gamma),
#           mc_value_slow(trajectory, gamma),
#           sum_discounted_rewards(trajectory, gamma)
#           )

import pytest
from rllib.dataset import Observation
from rllib.util.utilities import mc_value, _mc_value_slow, sum_discounted_rewards
import numpy as np
import scipy


@pytest.fixture(params=[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 2, 1, 0.2, 0.4]])
def trajectory(request):
    rewards = request.param
    trajectory_ = []
    for i in range(len(rewards)):
        observation = Observation(state=0, action=0, reward=rewards[i], next_state=0,
                                  done=False)
        trajectory_.append(observation)
    return trajectory_


def test_sanity():
    trajectory = [Observation(state=0, action=0, next_state=0, reward=1, done=0),
                  Observation(state=0, action=0, next_state=0, reward=0.5, done=0),
                  Observation(state=0, action=0, next_state=0, reward=2, done=0),
                  Observation(state=0, action=0, next_state=0, reward=-0.2, done=0)]
    for gamma in [1, 0.99, 0.9, 0]:
        cum_rewards = [1 + 0.5 * gamma + 2 * gamma ** 2 - 0.2 * gamma ** 3,
                       0.5 + 2 * gamma - 0.2 * gamma ** 2,
                       2 - 0.2 * gamma,
                       -0.2
                       ]
        scipy.allclose(cum_rewards, _mc_value_slow(trajectory, gamma))


def test_correctness(trajectory):
    rewards = []
    for observation in reversed(trajectory):
        rewards.append(observation.reward)
    cum_rewards = np.flip(np.cumsum(rewards))

    scipy.allclose(cum_rewards, _mc_value_slow(trajectory, 1))


def test_mc_value(trajectory):
    for gamma in [1]:
        assert scipy.allclose(mc_value(trajectory, gamma),
                              _mc_value_slow(trajectory, gamma))


def test_sum_discounted_rewards(trajectory):
    for gamma in [0.1, 0.9, 0.99, 1]:
        a = []
        for i in range(len(trajectory)):
            a.append(sum_discounted_rewards(trajectory[i:], gamma))

        assert scipy.allclose(mc_value(trajectory, gamma), a)

