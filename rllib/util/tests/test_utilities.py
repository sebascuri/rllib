import pytest
from rllib.util.utilities import discount_cumsum
import numpy as np
import scipy
import torch
import torch.testing


@pytest.fixture(params=[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 2, 1, 0.2, 0.4]])
def rewards(request):
    return request.param


@pytest.fixture(params=[1, 0.99, 0.9, 0])
def gamma(request):
    return request.param


def test_correctness(gamma):
    rewards = [1, 0.5, 2, -0.2]
    cum_rewards = [1 + 0.5 * gamma + 2 * gamma ** 2 - 0.2 * gamma ** 3,
                   0.5 + 2 * gamma - 0.2 * gamma ** 2,
                   2 - 0.2 * gamma,
                   -0.2
                   ]
    scipy.allclose(cum_rewards, discount_cumsum(np.array(rewards), gamma))

    torch.testing.assert_allclose(cum_rewards, discount_cumsum(torch.tensor(rewards),
                                                               gamma))


def test_sanity(rewards, gamma):
    np_returns = discount_cumsum(np.array(rewards), gamma)
    assert np_returns.shape == (len(rewards),)
    assert type(np_returns) is np.ndarray

    t_returns = discount_cumsum(torch.tensor(rewards).float(), gamma)
    assert t_returns.shape == torch.Size((len(rewards),))
    assert type(t_returns) is torch.Tensor

    torch.testing.assert_allclose(t_returns, np_returns)
