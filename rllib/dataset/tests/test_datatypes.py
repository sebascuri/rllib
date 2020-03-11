import numpy as np
import torch
from rllib.dataset.datatypes import Observation


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
        o = Observation(state, action, reward, next_state, done)
        assert o == Observation(state, action, reward, next_state, done)
        assert not o != Observation(state, action, reward, next_state, done)

    def test_inequality(self):
        state, action, reward, next_state, done = self.init()
        o = Observation(state, action, reward, next_state, done)
        assert o != Observation(state + 1, action, reward, next_state, done)
        assert not o == Observation(state + 1, action, reward, next_state, done)

        assert o != Observation(state, action + 1, reward, next_state, done)
        assert not o == Observation(state, action + 1, reward, next_state, done)

        assert o != Observation(state, action, reward + 1, next_state, done)
        assert not o == Observation(state, action, reward + 1, next_state, done)

        assert o != Observation(state, action, reward, next_state + 1, done)
        assert not o == Observation(state, action, reward, next_state + 1, done)

        assert o != Observation(state, action, reward, next_state, not done)
        assert not o == Observation(state, action, reward, next_state, not done)


class TestSARSAObservation(object):
    def init(self):
        state = np.random.randn(4)
        action = torch.randn(1)
        reward = 2.5
        next_state = np.random.randn(4)
        next_action = torch.randn(1)
        done = False
        return state, action, reward, next_state, done, next_action

    def test_equality(self):
        state, action, reward, nstate, done, n_action = self.init()
        o = Observation(state, action, reward, nstate, done, n_action)
        assert o == Observation(state, action, reward, nstate, done, n_action)
        assert not o != Observation(state, action, reward, nstate, done, n_action)

    def test_inequality(self):
        state, action, reward, nstate, done, nact = self.init()
        o = Observation(state, action, reward, nstate, done, nact)
        assert o != Observation(state + 1, action, reward, nstate, done, nact)
        assert not o == Observation(state + 1, action, reward, nstate, done, nact)

        assert o != Observation(state, action + 1, reward, nstate, done, nact)
        assert not o == Observation(state, action + 1, reward, nstate, done, nact)

        assert o != Observation(state, action, reward + 1, nstate, done, nact)
        assert not o == Observation(state, action, reward + 1, nstate, done, nact)

        assert o != Observation(state, action, reward, nstate + 1, done, nact)
        assert not o == Observation(state, action, reward, nstate + 1, done, nact)

        assert o != Observation(state, action, reward, nstate, not done, nact)
        assert not o == Observation(state, action, reward, nstate, not done, nact)

        assert o != Observation(state, action, reward, nstate, done, nact + 1)
        assert not o == Observation(state, action, reward, nstate, done, nact + 1)

    def test_type_inequality(self):
        state, action, reward, nstate, done, nact = self.init()
        o = Observation(state, action, reward, nstate, done, nact)

        assert not o == Observation(state, action, reward, nstate, done)
        assert not Observation(state, action, reward, nstate, done) == o