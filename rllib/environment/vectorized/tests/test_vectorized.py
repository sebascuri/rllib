import numpy as np
import torch
from gym.envs.classic_control import AcrobotEnv, CartPoleEnv, PendulumEnv

from rllib.environment.vectorized import (
    DiscreteVectorizedAcrobotEnv,
    DiscreteVectorizedCartPoleEnv,
    VectorizedAcrobotEnv,
    VectorizedCartPoleEnv,
    VectorizedPendulumEnv,
)


class TestAcrobot(object):
    @property
    def dims(self):
        return 4, 32

    @property
    def state_action(self):
        dim1, dim2 = self.dims
        state = np.random.randn(dim1, dim2, 4)
        action = np.random.randn(dim1, dim2, 1).clip(-1, 1)
        return state, action

    def test_np_shape(self):
        dim1, dim2 = self.dims

        env = VectorizedAcrobotEnv()
        state, action = self.state_action

        env.state = state
        obs, reward, done, _ = env.step(action)
        assert obs.shape == (dim1, dim2, 6)
        assert reward.shape == (dim1, dim2)
        assert done.shape == (dim1, dim2)

    def test_torch_shape(self):
        dim1, dim2 = self.dims

        env = VectorizedAcrobotEnv()
        state, action = self.state_action

        env.state = torch.tensor(state)
        obs, reward, done, _ = env.step(torch.tensor(action))
        assert obs.shape == (dim1, dim2, 6)
        assert reward.shape == (dim1, dim2)
        assert done.shape == (dim1, dim2)

    def test_torch_np_equality(self):
        env = VectorizedAcrobotEnv()
        state, action = self.state_action

        env.state = state
        np_obs, np_reward, np_done, _ = env.step(action)

        env.state = torch.tensor(state)
        t_obs, t_reward, t_done, _ = env.step(torch.tensor(action))

        np.testing.assert_almost_equal(np_obs, t_obs, 1e-6, 1e-6)
        np.testing.assert_almost_equal(np_reward, t_reward)
        np.testing.assert_almost_equal(np_done, t_done)

    def test_vectorized_original_equality(self):
        venv = VectorizedAcrobotEnv()
        state, action = self.state_action
        action = np.round(action)

        dim1, dim2 = self.dims

        venv.state = state
        vobs, vreward, vdone, _ = venv.step(action)

        env = AcrobotEnv()
        for i in range(dim1):
            for j in range(dim2):
                env.state = state[i, j]
                obs, reward, done, _ = env.step(1 + int(action[i, j, 0]))

                np.testing.assert_allclose(obs, vobs[i, j])
                np.testing.assert_allclose(reward, vreward[i, j])
                np.testing.assert_allclose(done, vdone[i, j])

    def test_discrete_vectorized_original_equality(self):
        venv = DiscreteVectorizedAcrobotEnv()
        state, action = self.state_action
        action = np.round(action) + 1

        dim1, dim2 = self.dims

        venv.state = state
        vobs, vreward, vdone, _ = venv.step(action)

        env = AcrobotEnv()
        for i in range(dim1):
            for j in range(dim2):
                env.state = state[i, j]
                obs, reward, done, _ = env.step(int(action[i, j, 0]))

                np.testing.assert_allclose(obs, vobs[i, j])
                np.testing.assert_allclose(reward, vreward[i, j])
                np.testing.assert_allclose(done, vdone[i, j])


class TestCartPole(object):
    @property
    def dims(self):
        return 4, 32

    @property
    def state_action(self):
        dim1, dim2 = self.dims
        state = np.random.randn(dim1, dim2, 4)
        action = np.random.randn(dim1, dim2, 1).clip(-1, 1)
        return state, action

    def test_np_shape(self):
        dim1, dim2 = self.dims

        env = VectorizedCartPoleEnv()
        state, action = self.state_action

        env.state = state
        obs, reward, done, _ = env.step(action)
        assert obs.shape == (dim1, dim2, 4)
        assert reward.shape == (dim1, dim2)
        assert done.shape == (dim1, dim2)

    def test_torch_shape(self):
        dim1, dim2 = self.dims

        env = VectorizedCartPoleEnv()
        state, action = self.state_action

        env.state = torch.tensor(state)
        obs, reward, done, _ = env.step(torch.tensor(action))
        assert obs.shape == (dim1, dim2, 4)
        assert reward.shape == (dim1, dim2)
        assert done.shape == (dim1, dim2)

    def test_torch_np_equality(self):
        env = VectorizedCartPoleEnv()
        state, action = self.state_action

        env.state = state
        np_obs, np_reward, np_done, _ = env.step(action)

        env.state = torch.tensor(state)
        t_obs, t_reward, t_done, _ = env.step(torch.tensor(action))

        np.testing.assert_almost_equal(np_obs, t_obs, 1e-6, 1e-6)
        np.testing.assert_almost_equal(np_reward, t_reward)
        np.testing.assert_almost_equal(np_done, t_done)

    def test_vectorized_original_equality(self):
        venv = VectorizedCartPoleEnv()
        state, action = self.state_action
        action = (action > 0).astype(int)

        dim1, dim2 = self.dims

        venv.state = state
        vobs, vreward, vdone, _ = venv.step(2 * action - 1)

        env = CartPoleEnv()
        for i in range(dim1):
            for j in range(dim2):
                env.reset()
                env.state = state[i, j]
                obs, reward, done, _ = env.step(action[i, j, 0])

                np.testing.assert_allclose(obs, vobs[i, j])
                np.testing.assert_allclose(reward, vreward[i, j])
                np.testing.assert_allclose(done, vdone[i, j])

    def test_discrete_vectorized_original_equality(self):
        venv = DiscreteVectorizedCartPoleEnv()
        state, action = self.state_action
        action = (action > 0).astype(int)

        dim1, dim2 = self.dims

        venv.state = state
        vobs, vreward, vdone, _ = venv.step(action)

        env = CartPoleEnv()
        for i in range(dim1):
            for j in range(dim2):
                env.reset()
                env.state = state[i, j]
                obs, reward, done, _ = env.step(int(action[i, j, 0]))

                np.testing.assert_allclose(obs, vobs[i, j])
                np.testing.assert_allclose(reward, vreward[i, j])
                np.testing.assert_allclose(done, vdone[i, j])


class TestPendulum(object):
    @property
    def dims(self):
        return 4, 32

    @property
    def state_action(self):
        dim1, dim2 = self.dims
        state = np.random.randn(dim1, dim2, 2)
        action = np.random.randn(dim1, dim2, 1)
        return state, action

    def test_np_shape(self):
        dim1, dim2 = self.dims

        env = VectorizedPendulumEnv()
        state, action = self.state_action

        env.state = state
        obs, reward, done, _ = env.step(action)
        assert obs.shape == (dim1, dim2, 3)
        assert reward.shape == (dim1, dim2)
        assert done.shape == (dim1, dim2)

    def test_torch_shape(self):
        dim1, dim2 = self.dims

        env = VectorizedPendulumEnv()
        state, action = self.state_action

        env.state = torch.tensor(state)
        obs, reward, done, _ = env.step(torch.tensor(action))
        assert obs.shape == (dim1, dim2, 3)
        assert reward.shape == (dim1, dim2)
        assert done.shape == (dim1, dim2)

    def test_torch_np_equality(self):
        env = VectorizedPendulumEnv()
        state, action = self.state_action

        env.state = state
        np_obs, np_reward, np_done, _ = env.step(action)

        env.state = torch.tensor(state)
        t_obs, t_reward, t_done, _ = env.step(torch.tensor(action))

        np.testing.assert_almost_equal(np_obs, t_obs, 1e-6, 1e-6)
        np.testing.assert_almost_equal(np_reward, t_reward)
        np.testing.assert_almost_equal(np_done, t_done)

    def test_vectorized_original_equality(self):
        venv = VectorizedPendulumEnv()
        state, action = self.state_action
        action = np.round(action)

        dim1, dim2 = self.dims

        venv.state = state
        vobs, vreward, vdone, _ = venv.step(action)

        env = PendulumEnv()
        for i in range(dim1):
            for j in range(dim2):
                env.state = state[i, j]
                obs, reward, done, _ = env.step(action[i, j])

                np.testing.assert_allclose(obs, vobs[i, j])
                np.testing.assert_allclose(reward, vreward[i, j])
                np.testing.assert_allclose(done, vdone[i, j])

    def test_set_state_np(self):
        venv = VectorizedPendulumEnv()
        state, action = self.state_action
        action = np.round(action)

        venv.state = state
        vobs, vreward, vdone, _ = venv.step(action)
        state = venv.state

        venv.set_state(vobs[2:])
        np.testing.assert_allclose(venv.state, state[2:])

    def test_set_state_torch(self):
        venv = VectorizedPendulumEnv()
        state, action = self.state_action
        action = np.round(action)

        venv.state = torch.tensor(state)
        vobs, vreward, vdone, _ = venv.step(torch.tensor(action))
        state = venv.state

        venv.set_state(vobs[2:])
        np.testing.assert_allclose(venv.state, state[2:])
