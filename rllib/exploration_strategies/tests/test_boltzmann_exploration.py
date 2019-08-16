from rllib.exploration_strategies import BoltzmannExploration
import torch
import torch.testing
from torch.distributions import MultivariateNormal, Categorical
import pytest
import numpy as np


@pytest.fixture(params=[0.0, 0.9])
def t_start(request):
    return request.param


def test_temperature(t_start):
    strategy = BoltzmannExploration(t_start)
    for t in range(100):
        assert strategy.temperature(t) == t_start
        assert strategy.temperature() == t_start

    strategy = BoltzmannExploration(t_start=t_start, t_end=0.1, t_decay=100)
    for t in range(100):
        assert strategy.temperature(t) == 0.1 + (t_start - 0.1) * np.exp(-t / 100)
        assert strategy.temperature() == t_start


def test_exact_discrete():
    strategy = BoltzmannExploration(t_start=0.)
    for t in range(100):
        logits = np.random.randn(3) ** 2
        action_distribution = Categorical(logits=torch.tensor(logits))
        assert strategy(action_distribution, t) == np.argmax(logits)


def test_exact_continuous():
    strategy = BoltzmannExploration(t_start=0.)
    for t in range(100):
        mean = torch.randn(4)
        action_distribution = MultivariateNormal(loc=mean,
                                                 covariance_matrix=torch.eye(4))
        torch.testing.assert_allclose(strategy(action_distribution, t), mean)


def test_call_discrete():
    strategy = BoltzmannExploration(t_start=0.9, t_end=0.1, t_decay=100)
    total = 0
    for t in range(100):
        logits = np.random.randn(3) ** 2
        action_distribution = Categorical(logits=torch.tensor(logits))
        action = strategy(action_distribution, t)
        assert action >= 0
        assert strategy(action_distribution, t) < 3
        if np.argmax(logits) != action:
            total += 1

    assert total


def test_call_continuous():
    strategy = BoltzmannExploration(t_start=0.9, t_end=0.1, t_decay=100)
    total = 0
    for t in range(100):
        mean = torch.randn(4)
        action_distribution = MultivariateNormal(loc=mean,
                                                 covariance_matrix=torch.eye(4))
        action = strategy(action_distribution, t)
        assert action.shape == (4,)
        if (mean.numpy() != action).any():
            total += 1

    assert total