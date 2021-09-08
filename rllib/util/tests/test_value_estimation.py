import numpy as np
import pytest
import scipy
import torch
import torch.testing

from rllib.dataset.datatypes import Observation
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.util.value_estimation import discount_cumsum, discount_sum, mc_return


class TestDiscountedCumSum(object):
    @pytest.fixture(params=[1, 0.99, 0.9, 0], scope="class")
    def gamma(self, request):
        return request.param

    @pytest.fixture(
        params=[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 2, 1, 0.2, 0.4]], scope="class"
    )
    def rewards(self, request):
        return request.param

    def test_correctness(self, gamma):
        rewards = [1, 0.5, 2, -0.2]
        cum_rewards = [
            1 + 0.5 * gamma + 2 * gamma ** 2 - 0.2 * gamma ** 3,
            0.5 + 2 * gamma - 0.2 * gamma ** 2,
            2 - 0.2 * gamma,
            -0.2,
        ]
        assert scipy.allclose(cum_rewards, discount_cumsum(np.array(rewards), gamma))

        torch.testing.assert_allclose(
            cum_rewards, discount_cumsum(torch.tensor(rewards), gamma)
        )

        torch.testing.assert_allclose(
            cum_rewards[0], discount_sum(torch.tensor(rewards), gamma)
        )

        rewards = np.tile(np.array(rewards), (5, 1))
        cum_rewards = np.tile(np.array(cum_rewards), (5, 1))

        assert scipy.allclose(cum_rewards, discount_cumsum(np.array(rewards), gamma))

        torch.testing.assert_allclose(
            torch.tensor(cum_rewards).float(),
            discount_cumsum(torch.tensor(rewards), gamma).float(),
        )

        torch.testing.assert_allclose(
            torch.tensor(cum_rewards[..., 0]).float(),
            discount_sum(torch.tensor(rewards, dtype=torch.get_default_dtype()), gamma),
        )

    def test_shape_and_type(self, rewards, gamma):
        np_returns = discount_cumsum(np.array(rewards), gamma)
        assert np_returns.shape == (len(rewards),)
        assert type(np_returns) is np.ndarray

        t_returns = discount_cumsum(
            torch.tensor(rewards, dtype=torch.get_default_dtype()), gamma
        )
        assert t_returns.shape == torch.Size((len(rewards),))
        assert type(t_returns) is torch.Tensor

        torch.testing.assert_allclose(t_returns, np_returns)


class TestMCReturn(object):
    @pytest.fixture(params=[1, 0.99, 0.9, 0.5, 0], scope="class")
    def gamma(self, request):
        return request.param

    @pytest.fixture(params=[True, False], scope="class")
    def value_function(self, request):
        if request:
            return lambda x: torch.tensor(0.01)
        else:
            return None

    @pytest.fixture(params=[1, 0.1, 0], scope="class")
    def entropy_reg(self, request):
        return request.param

    def test_correctness(self, gamma, value_function, entropy_reg):
        trajectory = [
            Observation(0, 0, reward=1, done=False, entropy=0.2).to_torch(),
            Observation(0, 0, reward=0.5, done=False, entropy=0.3).to_torch(),
            Observation(0, 0, reward=2, done=False, entropy=0.5).to_torch(),
            Observation(0, 0, reward=-0.2, done=False, entropy=-0.2).to_torch(),
        ]

        r0 = 1 + entropy_reg * 0.2
        r1 = 0.5 + entropy_reg * 0.3
        r2 = 2 + entropy_reg * 0.5
        r3 = -0.2 - entropy_reg * 0.2

        v = 0.01 if value_function is not None else 0

        reward = mc_return(
            stack_list_of_tuples(trajectory, -2),
            gamma,
            value_function=value_function,
            entropy_regularization=entropy_reg,
            reduction="min",
        )

        torch.testing.assert_allclose(
            reward,
            torch.tensor(
                [r0 + r1 * gamma + r2 * gamma ** 2 + r3 * gamma ** 3 + v * gamma ** 4]
            ),
        )
        assert (
            mc_return(
                Observation(state=0, reward=0).to_torch(),
                gamma,
                value_function,
                entropy_reg,
            )
            == 0
        )
