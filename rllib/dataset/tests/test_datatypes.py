import numpy as np
import torch
import torch.testing

from rllib.dataset.datatypes import RawObservation


class TestObservation(object):
    def init(self):
        state = np.random.randn(4)
        action = torch.randn(1)
        reward = 2
        next_state = np.random.randn(4)
        done = False
        return state, action, reward, next_state, done

    def test_equality(self):
        state, action, reward, next_state, done = self.init()
        o = RawObservation(state, action, reward, next_state, done).to_torch()
        for x, y in zip(
            o, RawObservation(state, action, reward, next_state, done).to_torch()
        ):
            torch.testing.assert_allclose(x, y)
        assert o is not RawObservation(state, action, reward, next_state, done)
