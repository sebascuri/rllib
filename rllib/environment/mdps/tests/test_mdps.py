import importlib

import numpy as np
import pytest

from rllib.environment import GymEnvironment


@pytest.fixture(params=[True, False])
def gym_env(request):
    """Request number of states."""
    return request.param


class MDPTest(object):
    """Basic class for MDP testing."""

    name: str

    def get_env(self, gym_env, **kwargs):
        """Get environment."""
        if gym_env:
            env = GymEnvironment(f"{self.name}-v0", **kwargs)
        else:
            module = importlib.import_module("rllib.environment.mdps")
            env = getattr(module, self.name)(**kwargs)
        return env


class TestBairdStar(MDPTest):
    """Test Baird Star environment."""

    name = "BairdStar"

    @pytest.fixture(params=[7, 14], scope="class")
    def num_states(self, request):
        """Request number of states."""
        return request.param

    def test_transitions(self, num_states):
        env = self.get_env(gym_env=False, num_states=num_states)
        assert env.num_actions == 2
        assert env.num_states == num_states
        env.check_transitions(env.transitions, env.num_states, env.num_actions)

    def test_initial_state(self, gym_env, num_states):
        env = self.get_env(gym_env=gym_env, num_states=num_states)
        for _ in range(10):
            state = env.reset()
            assert state in range(env.num_states)

    def test_interaction(self, gym_env, num_states):
        env = self.get_env(gym_env=gym_env, num_states=num_states)
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            if action == 0:  # Transit to state 0 when action 0 is applied.
                assert next_state == 0
            else:  # Transit to some other state when action 1 is applied.
                assert next_state in range(1, env.num_states)


class TestBoyanChain(MDPTest):
    """Test Boyan Chain environment."""

    name = "BoyanChain"

    @pytest.fixture(params=[7, 14], scope="class")
    def num_states(self, request):
        """Request number of states."""
        return request.param

    def test_transitions(self, num_states):
        env = self.get_env(gym_env=False, num_states=num_states)
        assert env.num_actions == 1
        assert env.num_states == num_states
        env.check_transitions(env.transitions, env.num_states, env.num_actions)

    def test_initial_state(self, gym_env):
        env = self.get_env(gym_env=gym_env)

        for _ in range(10):
            state = env.reset()
            assert state == 0

    def test_interaction(self, gym_env, num_states):
        env = self.get_env(gym_env=False, num_states=num_states)
        for _ in range(20):
            state = env.reset()
            for _ in range(env.num_states):
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)

                if state < num_states - 2:
                    assert next_state in [state + 1, state + 2]
                    assert reward == -3
                else:
                    assert next_state == num_states - 1
                    if state == num_states - 2:
                        assert reward == -2
                    else:
                        assert reward == 0

                state = next_state


class TestDoubleChainProblem(MDPTest):
    """Test Double Chain Problem."""

    name = "DoubleChainProblem"

    @pytest.fixture(params=[5, 10], scope="class")
    def chain_length(self, request):
        """Request number of states."""
        return request.param

    def test_transitions(self, chain_length):
        env = self.get_env(gym_env=False, chain_length=chain_length)
        assert env.num_actions == 2
        assert env.num_states == 2 * chain_length - 1
        env.check_transitions(env.transitions, env.num_states, env.num_actions)

    def test_initial_state(self, gym_env):
        env = self.get_env(gym_env=gym_env)
        for _ in range(10):
            state = env.reset()
            assert state == 0

    def test_interaction(self, gym_env, chain_length):
        env = self.get_env(gym_env=gym_env, chain_length=chain_length)

        env.state = 0
        next_state, reward, done, info = env.step(0)
        assert next_state == 1
        assert reward == 0

        env.state = 0
        next_state, reward, done, info = env.step(1)
        assert next_state == chain_length
        assert reward == 2

        state = env.reset()
        for _ in range(2 ** chain_length):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            if state != 0 and action == 1:  # Return to initial state.
                assert next_state == 0
                assert reward == 2
            elif state == 0 and action == 1:
                assert next_state == chain_length
                assert reward == 2
            else:  # Action == 0
                if state == chain_length - 1:
                    assert next_state == chain_length - 1
                    assert reward == 2 * chain_length
                elif state == 2 * chain_length - 2:
                    assert next_state == 2 * chain_length - 2
                    assert reward == chain_length
                else:
                    assert next_state == state + 1
                    assert reward == 0
            state = next_state


class TestEasyGridWorld(MDPTest):
    """Test Easy grid world."""

    name = "EasyGridWorld"

    @pytest.fixture(params=[0, 0.2], scope="class")
    def noise(self, request):
        """Request number of states."""
        return request.param

    @pytest.fixture(params=[4, 8], scope="class")
    def num_actions(self, request):
        """Request number of states."""
        return request.param

    def test_transitions(self, num_actions, noise):
        env = self.get_env(gym_env=False, num_actions=num_actions, noise=noise)
        assert env.num_actions == num_actions
        assert env.num_states == env.height * env.width
        env.check_transitions(env.transitions, env.num_states, env.num_actions)

    def test_initial_state(self):
        env = self.get_env(gym_env=False)
        for _ in range(10):
            state = env.reset()
            assert state in range(env.num_states)

    def test_interaction(self, num_actions, noise):
        env = self.get_env(gym_env=False, num_actions=num_actions, noise=noise)
        state = env.reset()
        for _ in range(200):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            assert action in range(env.num_actions)
            if np.all(env._state_to_grid(state) == [0, 1]):
                assert next_state == env._grid_to_state(np.array([env.height - 1, 1]))
                assert reward == 10
            elif np.all(env._state_to_grid(state) == [0, env.width - 2]):
                assert next_state == env._grid_to_state(
                    np.array([env.height // 2, env.width - 2])
                )
                assert reward == 5
            else:
                if state != next_state:
                    assert reward == 0

            state = next_state


class TestSingleChainProblem(MDPTest):
    """Test Single Chain Problem."""

    name = "SingleChainProblem"

    @pytest.fixture(params=[5, 10], scope="class")
    def chain_length(self, request):
        """Request number of states."""
        return request.param

    def test_transitions(self, chain_length):
        env = self.get_env(gym_env=False, chain_length=chain_length)
        assert env.num_actions == 2
        assert env.num_states == chain_length
        env.check_transitions(env.transitions, env.num_states, env.num_actions)

    def test_initial_state(self, gym_env):
        env = self.get_env(gym_env=gym_env)
        for _ in range(10):
            state = env.reset()
            assert state == 0

    def test_interaction(self, gym_env, chain_length):
        env = self.get_env(gym_env=gym_env, chain_length=chain_length)

        state = env.reset()
        for _ in range(2 ** chain_length):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)

            if action == 1:
                assert next_state == 0
                assert reward == 2
            else:
                if state < chain_length - 1:
                    assert next_state == state + 1
                    assert reward == 0
                else:
                    assert next_state == chain_length - 1
                    assert reward == 2 * chain_length

            state = next_state


class TestRandomMDP(MDPTest):
    """Test Random MDP."""

    name = "RandomMDP"

    @pytest.fixture(params=[20, 400], scope="class")
    def num_states(self, request):
        """Request number of states."""
        return request.param

    @pytest.fixture(params=[5, 10], scope="class")
    def num_actions(self, request):
        """Request number of states."""
        return request.param

    def test_transitions(self, num_states, num_actions):
        env = self.get_env(
            gym_env=False, num_states=num_states, num_actions=num_actions
        )

        assert env.num_actions == num_actions
        assert env.num_states == num_states
        env.check_transitions(env.transitions, env.num_states, env.num_actions)

    def test_initial_state(self, gym_env):
        env = self.get_env(gym_env=gym_env)

        for _ in range(10):
            state = env.reset()
            assert state in range(env.num_states)

    def test_interaction(self, gym_env):
        env = self.get_env(gym_env=gym_env)

        for _ in range(10):
            action = env.action_space.sample()
            env.step(action)


class TestTwoStateProblem(MDPTest):
    """Test Two State Problem."""

    name = "TwoStateProblem"

    def test_transitions(self):
        env = self.get_env(gym_env=False)
        assert env.num_actions == 2
        assert env.num_states == 2
        env.check_transitions(env.transitions, env.num_states, env.num_actions)

    def test_initial_state(self, gym_env):
        env = self.get_env(gym_env=gym_env)
        for _ in range(10):
            state = env.reset()
            assert state in [0, 1]

    def test_interaction(self, gym_env):
        env = self.get_env(gym_env=gym_env)

        state = env.reset()
        for _ in range(20):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)

            assert next_state == action
            if next_state == state:
                assert reward == state + 1
            else:
                assert reward == 0
            state = next_state
