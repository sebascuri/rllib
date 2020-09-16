import pytest
import torch
import torch.testing

from rllib.dataset.datatypes import Observation
from rllib.dataset.transforms import (
    ActionClipper,
    ActionNormalizer,
    ActionScaler,
    DeltaState,
    MeanFunction,
    RewardClipper,
    RewardScaler,
    StateNormalizer,
)
from rllib.dataset.utilities import stack_list_of_tuples


def get_observation(reward=None):
    return Observation(
        state=torch.randn(4),
        action=torch.randn(4),
        reward=reward if reward else torch.randn(1),
        next_state=torch.randn(4),
        done=False,
        state_scale_tril=torch.randn(4, 4),
        next_state_scale_tril=torch.randn(4, 4),
    ).to_torch()


@pytest.fixture
def trajectory():
    t = []
    for reward in [3.0, -2.0, 0.5]:
        t.append(get_observation(reward))
    return t


class MeanModel(torch.nn.Module):
    def forward(self, state, action):
        return 2 * state + action


class TestMeanFunction(object):
    def test_custom_mean(self, trajectory):
        transformer = MeanFunction(MeanModel())
        for observation in trajectory:
            obs = observation.clone()
            transformed_observation = transformer(observation)
            torch.testing.assert_allclose(transformed_observation.state, obs.state)
            torch.testing.assert_allclose(transformed_observation.action, obs.action)
            torch.testing.assert_allclose(transformed_observation.reward, obs.reward)
            assert transformed_observation.done == obs.done

            mean = 2 * obs.state + obs.action
            torch.testing.assert_allclose(
                transformed_observation.next_state, obs.next_state - mean
            )

    def test_call(self, trajectory):
        transformer = MeanFunction(DeltaState())
        for observation in trajectory:
            obs = observation.clone()
            transformed_observation = transformer(observation)
            torch.testing.assert_allclose(transformed_observation.state, obs.state)
            torch.testing.assert_allclose(transformed_observation.action, obs.action)
            torch.testing.assert_allclose(transformed_observation.reward, obs.reward)
            assert transformed_observation.done == obs.done

            mean = observation.state
            torch.testing.assert_allclose(
                transformed_observation.next_state, obs.next_state - mean
            )

    def test_inverse(self, trajectory):
        transformer = MeanFunction(DeltaState())
        for observation in trajectory:
            obs = observation.clone()
            inverse_observation = transformer.inverse(transformer(observation))
            for x, y in zip(obs, inverse_observation):
                torch.testing.assert_allclose(x, y)


class TestRewardClipper(object):
    @pytest.fixture(params=[0.0, None], scope="class")
    def minimum(self, request):
        return request.param

    @pytest.fixture(params=[1.0, None], scope="class")
    def maximum(self, request):
        return request.param

    def test_call(self, trajectory):
        transformer = RewardClipper(min_reward=0.0, max_reward=1.0)
        for observation in trajectory:
            obs = observation.clone()
            transformed_observation = transformer(observation)
            assert transformed_observation.reward >= 0.0
            assert transformed_observation.reward <= 1.0
            torch.testing.assert_allclose(transformed_observation.state, obs.state)
            torch.testing.assert_allclose(transformed_observation.action, obs.action)
            torch.testing.assert_allclose(
                transformed_observation.next_state, obs.next_state
            )
            assert transformed_observation.done == obs.done

            if 0 <= obs.reward <= 1:
                assert transformed_observation.reward == obs.reward
            elif obs.reward <= 0:
                assert transformed_observation.reward == 0.0
            elif obs.reward >= 1.0:
                assert transformed_observation.reward == 1.0

    def test_inverse(self, trajectory):
        transformer = RewardClipper(min_reward=0.0, max_reward=1.0)
        for observation in trajectory:
            obs = observation.clone()
            transformed_observation = transformer(observation)
            inverse_observation = transformer.inverse(transformed_observation)

            torch.testing.assert_allclose(transformed_observation.state, obs.state)
            torch.testing.assert_allclose(transformed_observation.action, obs.action)
            torch.testing.assert_allclose(
                transformed_observation.next_state, obs.next_state
            )
            assert transformed_observation.done == obs.done
            torch.testing.assert_allclose(
                inverse_observation.reward, transformed_observation.reward
            )


class TestActionClipper(object):
    @pytest.fixture(params=[0.0, None], scope="class")
    def minimum(self, request):
        return request.param

    @pytest.fixture(params=[1.0, None], scope="class")
    def maximum(self, request):
        return request.param

    def test_call(self, trajectory):
        transformer = ActionClipper(min_action=0.0, max_action=1.0)
        for observation in trajectory:
            obs = observation.clone()
            transformed_observation = transformer(observation)
            assert (transformed_observation.action >= 0.0).all()
            assert (transformed_observation.action <= 1.0).all()
            torch.testing.assert_allclose(transformed_observation.state, obs.state)
            torch.testing.assert_allclose(transformed_observation.reward, obs.reward)
            torch.testing.assert_allclose(
                transformed_observation.next_state, obs.next_state
            )
            assert transformed_observation.done == obs.done

    def test_inverse(self, trajectory):
        transformer = ActionClipper(min_action=-1000.0, max_action=1000.0)

        for observation in trajectory:
            obs = observation.clone()
            transformed_observation = transformer(observation)
            inverse_observation = transformer.inverse(transformed_observation)

            torch.testing.assert_allclose(transformed_observation.state, obs.state)
            torch.testing.assert_allclose(transformed_observation.action, obs.action)
            torch.testing.assert_allclose(
                transformed_observation.next_state, obs.next_state
            )
            assert transformed_observation.done == obs.done
            torch.testing.assert_allclose(
                inverse_observation.reward, transformed_observation.reward
            )


class TestRewardScaler(object):
    def test_call(self, trajectory):
        transformer = RewardScaler(scale=5.0)
        for observation in trajectory:
            obs = observation.clone()
            transformed_observation = transformer(observation)
            torch.testing.assert_allclose(transformed_observation.state, obs.state)
            torch.testing.assert_allclose(transformed_observation.action, obs.action)
            torch.testing.assert_allclose(
                transformed_observation.next_state, obs.next_state
            )
            assert transformed_observation.done == obs.done
            torch.testing.assert_allclose(
                transformed_observation.reward, obs.reward / 5
            )

    def test_inverse(self, trajectory):
        transformer = RewardScaler(scale=5.0)
        for observation in trajectory:
            obs = observation.clone()
            transformed_observation = transformer(observation)
            inverse_observation = transformer.inverse(transformed_observation)

            torch.testing.assert_allclose(transformed_observation.state, obs.state)
            torch.testing.assert_allclose(transformed_observation.action, obs.action)
            torch.testing.assert_allclose(
                transformed_observation.next_state, obs.next_state
            )
            assert transformed_observation.done == obs.done
            torch.testing.assert_allclose(inverse_observation.reward, obs.reward)


class TestActionScaler(object):
    def test_call(self, trajectory):
        transformer = ActionScaler(scale=5.0)
        for observation in trajectory:
            obs = observation.clone()
            transformed_observation = transformer(observation)
            torch.testing.assert_allclose(transformed_observation.state, obs.state)
            torch.testing.assert_allclose(transformed_observation.reward, obs.reward)
            torch.testing.assert_allclose(
                transformed_observation.next_state, obs.next_state
            )
            assert transformed_observation.done == obs.done
            torch.testing.assert_allclose(
                transformed_observation.action, obs.action / 5
            )

    def test_inverse(self, trajectory):
        transformer = RewardScaler(scale=5.0)
        for observation in trajectory:
            obs = observation.clone()
            transformed_observation = transformer(observation)
            inverse_observation = transformer.inverse(transformed_observation)

            torch.testing.assert_allclose(transformed_observation.state, obs.state)
            torch.testing.assert_allclose(transformed_observation.action, obs.action)
            torch.testing.assert_allclose(
                transformed_observation.next_state, obs.next_state
            )
            assert transformed_observation.done == obs.done
            torch.testing.assert_allclose(inverse_observation.reward, obs.reward)


class TestActionNormalize(object):
    @pytest.fixture(params=[True, False], scope="class")
    def preserve_origin(self, request):
        return request.param

    def test_update(self, trajectory, preserve_origin):
        transformer = ActionNormalizer(preserve_origin)

        trajectory = stack_list_of_tuples(trajectory)

        mean = torch.mean(trajectory.action, 0)
        var = torch.var(trajectory.action, 0)

        transformer.update(trajectory)
        torch.testing.assert_allclose(transformer._normalizer.mean, mean)
        torch.testing.assert_allclose(transformer._normalizer.variance, var)

    def test_call(self, trajectory, preserve_origin):
        transformer = ActionNormalizer(preserve_origin)

        trajectory = stack_list_of_tuples(trajectory)
        transformer.update(trajectory)
        observation = get_observation()
        obs = observation.clone()
        transformed = transformer(observation)

        if preserve_origin:
            mean = 0
            scale = torch.sqrt(
                transformer._normalizer.variance + transformer._normalizer.mean ** 2
            )
        else:
            mean = transformer._normalizer.mean
            scale = torch.sqrt(transformer._normalizer.variance)
        torch.testing.assert_allclose(transformed.action, (obs.action - mean) / scale)

        torch.testing.assert_allclose(transformed.state, obs.state)
        torch.testing.assert_allclose(transformed.reward, obs.reward)
        torch.testing.assert_allclose(transformed.next_state, obs.next_state)
        assert transformed.done == obs.done

    def test_inverse(self, trajectory, preserve_origin):
        transformer = ActionNormalizer(preserve_origin)
        trajectory = stack_list_of_tuples(trajectory)
        transformer.update(trajectory)

        observation = get_observation()
        obs = observation.clone()
        inverse_observation = transformer.inverse(transformer(observation))
        for x, y in zip(obs, inverse_observation):
            if x.shape == y.shape:
                torch.testing.assert_allclose(x, y)


class TestStateNormalize(object):
    @pytest.fixture(params=[True, False], scope="class")
    def preserve_origin(self, request):
        return request.param

    def test_update(self, trajectory, preserve_origin):
        transformer = StateNormalizer(preserve_origin)

        trajectory = stack_list_of_tuples(trajectory)

        mean = torch.mean(trajectory.state, 0)
        var = torch.var(trajectory.state, 0)

        transformer.update(trajectory)
        torch.testing.assert_allclose(transformer._normalizer.mean, mean)
        torch.testing.assert_allclose(transformer._normalizer.variance, var)

    def test_call(self, trajectory, preserve_origin):
        transformer = StateNormalizer(preserve_origin)
        trajectory = stack_list_of_tuples(trajectory)

        transformer.update(trajectory)
        observation = get_observation()
        obs = observation.clone()
        transformed = transformer(observation)

        if preserve_origin:
            mean = 0
            scale = torch.sqrt(
                transformer._normalizer.variance + transformer._normalizer.mean ** 2
            )
        else:
            mean = transformer._normalizer.mean
            scale = torch.sqrt(transformer._normalizer.variance)
        torch.testing.assert_allclose(transformed.state, (obs.state - mean) / scale)
        torch.testing.assert_allclose(transformed.next_state, obs.next_state)

        torch.testing.assert_allclose(transformed.action, obs.action)
        torch.testing.assert_allclose(transformed.reward, obs.reward)
        assert transformed.done == obs.done

    def test_inverse(self, trajectory, preserve_origin):
        transformer = StateNormalizer(preserve_origin)

        trajectory = stack_list_of_tuples(trajectory)

        transformer.update(trajectory)

        observation = get_observation()
        obs = observation.clone()
        inverse_observation = transformer.inverse(transformer(observation))
        for x, y in zip(obs, inverse_observation):
            if x.shape == y.shape:
                torch.testing.assert_allclose(x, y)
