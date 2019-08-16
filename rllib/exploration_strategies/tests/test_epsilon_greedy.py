from rllib.exploration_strategies import EpsGreedy
import torch
import torch.testing
from torch.distributions import MultivariateNormal, Categorical, HalfCauchy
import pytest
import numpy as np


@pytest.fixture(params=[0.0, 0.9])
def eps_start(request):
    return request.param


def test_epsilon(eps_start):
    strategy = EpsGreedy(eps_start=eps_start)
    for t in range(100):
        assert strategy.epsilon(t) == eps_start
        assert strategy.epsilon() == eps_start

    strategy = EpsGreedy(eps_start=eps_start, eps_end=0.1, eps_decay=100)
    for t in range(100):
        assert strategy.epsilon(t) == 0.1 + (eps_start - 0.1) * np.exp(-t / 100)
        assert strategy.epsilon() == eps_start


def test_exact_discrete():
    strategy = EpsGreedy(eps_start=0.)
    for t in range(100):
        logits = np.random.randn(3) ** 2
        action_distribution = Categorical(logits=torch.tensor(logits))
        assert strategy(action_distribution, t) == np.argmax(logits)


def test_exact_continuous():
    strategy = EpsGreedy(eps_start=0.)
    for t in range(100):
        mean = torch.randn(4)
        action_distribution = MultivariateNormal(loc=mean,
                                                 covariance_matrix=torch.eye(4))
        torch.testing.assert_allclose(strategy(action_distribution, t), mean)


def test_call_discrete():
    strategy = EpsGreedy(eps_start=0.9, eps_end=0.1, eps_decay=100)
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
    strategy = EpsGreedy(eps_start=0.9, eps_end=0.1, eps_decay=100)
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


def test_wrong_distribution():
    strategy = EpsGreedy(eps_start=0)
    action_distribution = HalfCauchy(scale=1)
    with pytest.raises(NotImplementedError):
        strategy(action_distribution)