import numpy as np
from rllib.dataset import Observation
import pytest


@pytest.fixture()
def observation():
    """Init tests"""
    state = np.array([0., 0., 1.])
    action = np.array([1])
    next_state = np.array([0.1, 0.1, 1.2])
    reward = np.dot(state, state) + np.dot(action, action)
    return Observation(state, action, reward, next_state)


def test_init(observation):
    """Test Initialization."""
    assert True


def test_state(observation):
    assert np.all(np.array([0., 0., 1.]) == observation.state)


def test_action(observation):
    assert np.all(np.array([1]) == observation.action)


def test_reward(observation):
    assert 2.0 == observation.reward


def test_next_state(observation):
    assert np.all(np.array([0.1, 0.1, 1.2]) == observation.next_state)


def test_state_dim(observation):
    assert 3 == observation.state_dim


def test_action_dim(observation):
    assert 1 == observation.action_dim
