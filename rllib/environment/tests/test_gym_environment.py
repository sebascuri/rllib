import numpy as np
import torch

from rllib.environment import GymEnvironment


class TestPendulumGymEnvironment(object):
    @property
    def dims(self):
        return 4, 32

    @property
    def state_action(self):
        dim1, dim2 = self.dims
        state = 1 - 2 * np.random.rand(dim1, dim2, 3)
        action = np.random.randn(dim1, dim2, 1)
        return state, action

    def test_set_state_np(self):
        env = GymEnvironment("VPendulum-v0")
        env.reset()

        state, action = self.state_action
        action = np.round(action)

        env.state = state
        obs, _, _, _ = env.step(action)
        state = env.state
        np.testing.assert_allclose(obs, state)

    def test_set_state_torch(self):
        env = GymEnvironment("VPendulum-v0")
        env.reset()

        state, action = self.state_action
        action = np.round(action)

        env.state = torch.tensor(state)
        obs, _, _, _ = env.step(torch.tensor(action))
        state = env.state

        np.testing.assert_allclose(obs, state)
