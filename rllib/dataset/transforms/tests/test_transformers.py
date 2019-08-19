from rllib.dataset.transforms import *
from rllib.dataset import Observation, stack_list_of_tuples
from rllib.dataset.transforms.utilities import *
import pytest
import numpy as np
import torch


def _trajectory(backend):
    t = []
    for reward in [3, -2, 0.5]:
        t.append(get_observation(backend, reward))
    return t


def get_observation(backend, reward=None):
    if backend is np:
        rand = np.random.randn
    else:
        rand = torch.randn

    return Observation(state=rand(4),
                       action=rand(4),
                       reward=reward if reward else rand(1),
                       next_state=rand(4),
                       done=False)


@pytest.fixture(params=[np, torch])
def trajectory(request):
    return _trajectory(request.param)


class TestMeanFunction(object):
    def test_call(self, trajectory):
        transformer = MeanFunction(lambda state, action: 2 * state + action)
        for observation in trajectory:
            backend = get_backend(observation.state)
            transformed_observation = transformer(observation)
            backend.testing.assert_allclose(transformed_observation.state,
                                            observation.state)
            backend.testing.assert_allclose(transformed_observation.action,
                                            observation.action)
            backend.testing.assert_allclose(transformed_observation.reward,
                                            observation.reward)
            backend.testing.assert_allclose(transformed_observation.done,
                                            observation.done)

            mean = 2 * observation.state + observation.action
            backend.testing.assert_allclose(transformed_observation.next_state,
                                            observation.next_state - mean)

        transformer.update(trajectory)

    def test_inverse(self, trajectory):
        transformer = MeanFunction(lambda state, action: 2 * state + action)
        for observation in trajectory:
            inverse_observation = transformer.inverse(transformer(observation))
            assert observation == inverse_observation
            assert observation is not inverse_observation


class TestRewardClipper(object):
    @pytest.fixture(params=[0., None], scope="class")
    def minimum(self, request):
        return request.param

    @pytest.fixture(params=[1., None], scope="class")
    def maximum(self, request):
        return request.param

    def test_assignment(self, minimum, maximum):
        if minimum is None and maximum is None:
            transformer = RewardClipper()
        elif minimum is None and maximum is not None:
            transformer = RewardClipper(max_reward=maximum)
        elif minimum is not None and maximum is None:
            transformer = RewardClipper(min_reward=minimum)
        else:
            transformer = RewardClipper(min_reward=minimum, max_reward=maximum)

        assert transformer._min_reward == 0.
        assert transformer._max_reward == 1.

    def test_call(self, trajectory):
        transformer = RewardClipper(min_reward=0., max_reward=1.)
        for observation in trajectory:
            transformed_observation = transformer(observation)
            assert transformed_observation.reward >= 0.
            assert transformed_observation.reward <= 1.
            backend = get_backend(observation.state)
            backend.testing.assert_allclose(transformed_observation.state,
                                            observation.state)
            backend.testing.assert_allclose(transformed_observation.action,
                                            observation.action)
            backend.testing.assert_allclose(transformed_observation.next_state,
                                            observation.next_state)
            backend.testing.assert_allclose(transformed_observation.done,
                                            observation.done)

            if 0 <= observation.reward <= 1:
                assert transformed_observation.reward == observation.reward
            elif observation.reward <= 0:
                assert transformed_observation.reward == 0.
            elif observation.reward >= 1.:
                assert transformed_observation.reward == 1.

    def test_inverse(self, trajectory):
        transformer = RewardClipper(min_reward=0., max_reward=1.)
        for observation in trajectory:
            transformed_observation = transformer(observation)
            inverse_observation = transformer.inverse(transformed_observation)

            backend = get_backend(observation.state)
            backend.testing.assert_allclose(transformed_observation.state,
                                            observation.state)
            backend.testing.assert_allclose(transformed_observation.action,
                                            observation.action)
            backend.testing.assert_allclose(transformed_observation.next_state,
                                            observation.next_state)
            backend.testing.assert_allclose(transformed_observation.done,
                                            observation.done)

            backend.testing.assert_allclose(inverse_observation.reward,
                                            transformed_observation.reward)


class TestActionNormalize(object):
    @pytest.fixture(params=[True, False], scope="class")
    def preserve_origin(self, request):
        return request.param

    def test_update(self, trajectory, preserve_origin):
        transformer = ActionNormalizer(preserve_origin)
        backend = get_backend(trajectory[0].state)
        if backend == torch:
            return

        trajectory = stack_list_of_tuples(trajectory, backend=backend)

        mean = backend.mean(trajectory.action, axis=0)
        var = backend.var(trajectory.action, axis=0)

        transformer.update(trajectory)
        backend.testing.assert_allclose(transformer._normalizer._mean, mean)
        backend.testing.assert_allclose(transformer._normalizer._variance, var)

    def test_call(self, trajectory, preserve_origin):
        backend = get_backend(trajectory[0].state)
        if backend == torch:
            return

        transformer = ActionNormalizer(preserve_origin)

        trajectory = stack_list_of_tuples(trajectory)
        transformer.update(trajectory)
        observation = get_observation(backend)
        transformed = transformer(observation)

        if preserve_origin:
            mean = 0
            scale = backend.sqrt(transformer._normalizer._variance
                                 + transformer._normalizer._mean ** 2)
        else:
            mean = transformer._normalizer._mean
            scale = backend.sqrt(transformer._normalizer._variance)
        backend.testing.assert_allclose(transformed.action,
                                        (observation.action - mean) / scale)

        backend.testing.assert_allclose(transformed.state, observation.state)
        backend.testing.assert_allclose(transformed.reward, observation.reward)
        backend.testing.assert_allclose(transformed.next_state, observation.next_state)
        backend.testing.assert_allclose(transformed.done, observation.done)

    def test_inverse(self, trajectory, preserve_origin):
        backend = get_backend(trajectory[0].state)
        if backend == torch:
            return

        transformer = ActionNormalizer(preserve_origin)
        trajectory = stack_list_of_tuples(trajectory)
        transformer.update(trajectory)

        observation = get_observation(backend)
        inverse_observation = transformer.inverse(transformer(observation))
        assert observation == inverse_observation
        assert observation is not inverse_observation


class TestStateNormalize(object):
    @pytest.fixture(params=[True, False], scope="class")
    def preserve_origin(self, request):
        return request.param

    def test_batch_update(self, trajectory, preserve_origin):
        transformer = StateNormalizer(preserve_origin)
        backend = get_backend(trajectory[0].state)
        if backend == torch:
            return

        trajectory = stack_list_of_tuples(trajectory)
        mean = np.mean(trajectory.state, axis=0)
        var = np.var(trajectory.state, axis=0)

        transformer.update(trajectory)
        np.testing.assert_allclose(transformer._normalizer._mean, mean)
        np.testing.assert_allclose(transformer._normalizer._variance, var)

    def test_call(self, trajectory, preserve_origin):
        transformer = StateNormalizer(preserve_origin)
        backend = get_backend(trajectory[0].state)
        if backend == torch:
            return

        trajectory = stack_list_of_tuples(trajectory)
        transformer.update(trajectory)
        observation = get_observation(backend)
        transformed = transformer(observation)

        if preserve_origin:
            mean = 0
            scale = backend.sqrt(transformer._normalizer._variance
                                 + transformer._normalizer._mean ** 2)
        else:
            mean = transformer._normalizer._mean
            scale = backend.sqrt(transformer._normalizer._variance)
        backend.testing.assert_allclose(transformed.state,
                                        (observation.state - mean) / scale)
        backend.testing.assert_allclose(transformed.next_state,
                                        (observation.next_state - mean) / scale)

        backend.testing.assert_allclose(transformed.action, observation.action)
        backend.testing.assert_allclose(transformed.reward, observation.reward)
        backend.testing.assert_allclose(transformed.done, observation.done)

    def test_inverse(self, trajectory, preserve_origin):
        transformer = StateNormalizer(preserve_origin)
        backend = get_backend(trajectory[0].state)
        if backend == torch:
            return

        trajectory = stack_list_of_tuples(trajectory)
        transformer.update(trajectory)

        observation = get_observation(backend)
        inverse_observation = transformer.inverse(transformer(observation))
        assert observation == inverse_observation
        assert observation is not inverse_observation