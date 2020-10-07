import numpy as np
import pytest

from rllib.dataset import ExperienceReplay
from rllib.dataset.datatypes import Observation
from rllib.dataset.transforms import (
    ActionNormalizer,
    MeanFunction,
    RewardClipper,
    StateNormalizer,
)
from rllib.environment import GymEnvironment
from rllib.util.rollout import step_env


def create_er_from_episodes(discrete, max_len, num_steps, num_episodes, episode_length):
    """Rollout an environment and return an Experience Replay Buffer."""

    if discrete:
        env = GymEnvironment("NChain-v0")
        transformations = []
    else:
        env = GymEnvironment("Pendulum-v0")
        transformations = [
            MeanFunction(lambda state_, action_: state_),
            StateNormalizer(),
            ActionNormalizer(),
            RewardClipper(),
        ]

    memory = ExperienceReplay(
        max_len, transformations=transformations, num_steps=num_steps
    )

    for _ in range(num_episodes):
        state = env.reset()
        for _ in range(episode_length):
            action = env.action_space.sample()  # sample a random action.
            observation, state, done, info = step_env(
                env, state, action, action_scale=1.0
            )
            memory.append(observation)
        memory.end_episode()

    return memory


def create_er_from_transitions(
    discrete, dim_state, dim_action, max_len, num_steps, num_transitions
):
    """Create a memory with `num_transitions' transitions."""
    if discrete:
        num_states, num_actions = dim_state, dim_action
        dim_state, dim_action = (), ()
    else:
        num_states, num_actions = -1, -1
        dim_state, dim_action = (dim_state,), (dim_action,)

    memory = ExperienceReplay(max_len, num_steps=num_steps)
    for _ in range(num_transitions):
        observation = Observation.random_example(
            dim_state=dim_state,
            dim_action=dim_action,
            num_states=num_states,
            num_actions=num_actions,
        )
        memory.append(observation)
    return memory


class TestExperienceReplay(object):
    """Test experience replay class."""

    @pytest.fixture(scope="class", params=[True, False])
    def discrete(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1, 4])
    def dim_state(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1, 4])
    def dim_action(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[100, 20000])
    def max_len(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[0, 1, 5])
    def num_steps(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1, 64])
    def batch_size(self, request):
        return request.param

    def _test_sample_batch(self, memory, batch_size, num_steps):
        observation, idx, weight = memory.sample_batch(batch_size=batch_size)
        for attribute in observation:
            if num_steps == 0:
                assert attribute.shape[:2] == (batch_size, 1)
            else:
                assert attribute.shape[:2] == (batch_size, num_steps)

        assert idx.shape == (batch_size,)
        assert weight.shape == (batch_size,)

    def test_sample_batch_from_episode(self, discrete, max_len, num_steps, batch_size):
        num_episodes = 3
        episode_length = 200
        memory = create_er_from_episodes(
            discrete, max_len, num_steps, num_episodes, episode_length
        )
        self._test_sample_batch(memory, batch_size, num_steps)

    def test_sample_batch_from_transitions(
        self, discrete, dim_state, dim_action, max_len, num_steps, batch_size
    ):
        memory = create_er_from_transitions(
            discrete, dim_state, dim_action, max_len, num_steps, 200
        )

        self._test_sample_batch(memory, batch_size, num_steps)

    def test_reset(self, discrete, max_len, num_steps):
        num_episodes = 3
        episode_length = 200
        memory = create_er_from_episodes(
            discrete, max_len, num_steps, num_episodes, episode_length
        )
        assert memory.data_count == num_episodes * (episode_length + num_steps)
        if num_steps == 0:
            assert len(memory.valid_indexes) == min(max_len, memory.data_count)
        else:
            assert len(memory.valid_indexes) > 0
            assert len(memory.valid_indexes) < len(memory)

            assert memory.ptr != 0
        assert not np.all(memory.memory == np.full((max_len,), None))

        memory.reset()
        assert memory.data_count == 0
        assert len(memory.valid_indexes) == 0
        assert memory.ptr == 0
        assert np.all(memory.memory == np.full((max_len,), None))

    def test_end_episode(self, discrete, dim_state, dim_action, max_len, num_steps):
        num_transitions = 200
        memory = create_er_from_transitions(
            discrete, dim_state, dim_action, max_len, num_steps, num_transitions
        )
        ptr = memory.ptr
        memory.end_episode()
        assert ptr + num_steps == memory.ptr
        for i in range(num_steps):
            assert memory.valid[memory.ptr - i - 1] == 0

    def test_append_invalid(self, discrete, dim_state, dim_action, max_len, num_steps):
        num_transitions = 200
        memory = create_er_from_transitions(
            discrete, dim_state, dim_action, max_len, num_steps, num_transitions
        )
        memory.append_invalid()
        assert memory.valid[(memory.ptr - 1) % max_len] == 0
        assert memory.valid[(memory.ptr - 2) % max_len] == 1

    def test_append(self, discrete, dim_state, dim_action, max_len, num_steps):
        num_transitions = 200
        memory = create_er_from_transitions(
            discrete, dim_state, dim_action, max_len, num_steps, num_transitions
        )
        if discrete:
            num_states, num_actions = dim_state, dim_action
            dim_state, dim_action = (), ()
        else:
            num_states, num_actions = -1, -1
            dim_state, dim_action = (dim_state,), (dim_action,)
        observation = Observation.random_example(
            dim_state=dim_state,
            dim_action=dim_action,
            num_states=num_states,
            num_actions=num_actions,
        )

        memory.append(observation)
        assert memory.valid[(memory.ptr - 1) % max_len] == 1
        assert memory.valid[(memory.ptr - 2) % max_len] == 1
        for i in range(num_steps):
            assert memory.valid[(memory.ptr + i) % max_len] == 0
        assert memory.memory[(memory.ptr - 1) % max_len] is not observation

    def test_len(self, discrete, dim_state, dim_action, max_len, num_steps):
        num_transitions = 200
        memory = create_er_from_transitions(
            discrete, dim_state, dim_action, max_len, num_steps, num_transitions
        )
        assert len(memory) == min(max_len, num_transitions)

        num_episodes = 3
        episode_length = 200
        memory = create_er_from_episodes(
            discrete, max_len, num_steps, num_episodes, episode_length
        )
        if num_steps == 0:
            assert len(memory) == len(memory.valid_indexes)
        else:
            assert len(memory) > len(memory.valid_indexes)
        assert len(memory) == min(max_len, num_episodes * (episode_length + num_steps))

    def test_get_item(self, discrete, max_len, num_steps):
        num_episodes = 3
        episode_length = 200
        memory = create_er_from_episodes(
            discrete, max_len, num_steps, num_episodes, episode_length
        )
        memory.end_episode()

        observation, idx, weight = memory[0]
        for attribute in Observation(**observation):
            assert attribute.shape[0] == max(1, num_steps)
            assert idx == 0
            assert weight == 1.0

        for i in range(len(memory)):
            observation, idx, weight = memory[i]
            for attribute in Observation(**observation):
                assert attribute.shape[0] == max(1, num_steps)
                if memory.valid[i]:
                    assert idx == i
                else:
                    assert idx != i
                assert weight == 1.0

        i = np.random.choice(memory.valid_indexes).item()
        observation, idx, weight = memory[i]
        for attribute in Observation(**observation):
            assert attribute.shape[0] == max(1, num_steps)
            assert idx == i
            assert weight == 1.0

    def test_is_full(self, discrete, dim_state, dim_action, max_len, num_steps):
        num_transitions = 98
        memory = create_er_from_transitions(
            discrete, dim_state, dim_action, max_len, num_steps, num_transitions
        )
        if num_transitions >= max_len:
            assert memory.is_full
        else:
            assert not memory.is_full

        memory.end_episode()

        if num_transitions + num_steps >= max_len:
            assert memory.is_full
        else:
            assert not memory.is_full

    def test_all_data(self, discrete, max_len, num_steps):
        num_episodes = 3
        episode_length = 200
        memory = create_er_from_episodes(
            discrete, max_len, num_steps, num_episodes, episode_length
        )
        observation = memory.all_data
        len_valid_data = len(memory.valid_indexes)
        if memory.is_full:
            for attribute in observation:
                assert attribute.shape[0] == max_len - num_steps
                assert attribute.shape[0] == len_valid_data
        else:
            for attribute in observation:
                assert attribute.shape[0] == num_episodes * episode_length
                assert attribute.shape[0] == len_valid_data

    def test_num_steps(self, discrete, max_len, num_steps):
        num_episodes = 3
        episode_length = 200
        memory = create_er_from_episodes(
            discrete, max_len, num_steps, num_episodes, episode_length
        )
        self._test_sample_batch(memory, 10, num_steps)
        assert memory.num_steps == num_steps

        memory.num_steps = 10
        assert memory.num_steps == 10
        self._test_sample_batch(memory, 10, 10)

        memory.num_steps = 2
        assert memory.num_steps == 2
        self._test_sample_batch(memory, 10, 2)

    def test_append_error(self):
        memory = ExperienceReplay(max_len=100)
        with pytest.raises(TypeError):
            memory.append((1, 2, 3, 4, 5))

    def test_valid_indexes(self, discrete, max_len, num_steps):
        num_episodes = 3
        episode_length = 200
        memory = create_er_from_episodes(
            discrete, max_len, num_steps, num_episodes, episode_length
        )
        for i in memory.valid_indexes:
            assert memory.valid[i] == 1
        if not memory.is_full:
            assert len(memory.valid_indexes) == num_episodes * episode_length
        else:
            assert len(memory.valid_indexes) == max_len - num_steps

    def test_iter(self, discrete, max_len, num_steps):
        num_episodes = 3
        episode_length = 200
        memory = create_er_from_episodes(
            discrete, max_len, num_steps, num_episodes, episode_length
        )

        for idx, (observation, idx_, weight) in enumerate(memory):
            if idx >= len(memory):
                continue

            if memory.valid[idx] == 1:
                assert idx == idx_
            else:
                assert idx != idx_

            assert weight == 1.0
            for attribute in Observation(**observation):
                assert attribute.shape[0] == max(1, num_steps)
