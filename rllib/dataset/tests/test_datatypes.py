import numpy as np
import pytest
import torch
import torch.testing

from rllib.dataset.datatypes import RawObservation


class TestObservation(object):
    @pytest.fixture(params=["zero", "random", "nan", "dummy"], scope="class")
    def kind(self, request):
        return request.param

    @pytest.fixture(params=[True, False], scope="class")
    def discrete(self, request):
        return request.param

    @pytest.fixture(params=[2, 4], scope="class")
    def dim_state(self, request):
        return request.param

    @pytest.fixture(params=[2, 4], scope="class")
    def dim_action(self, request):
        return request.param

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

    def test_example(self, discrete, dim_state, dim_action, kind):
        if discrete:
            num_states, num_actions = dim_state, dim_action
            dim_state, dim_action = 1, 1
        else:
            num_states, num_actions = -1, -1

        if kind == "nan":
            o = RawObservation.nan_example(
                dim_state=dim_state,
                dim_action=dim_action,
                num_states=num_states,
                num_actions=num_actions,
            )
        elif kind == "zero":
            o = RawObservation.zero_example(
                dim_state=dim_state,
                dim_action=dim_action,
                num_states=num_states,
                num_actions=num_actions,
            )
        elif kind == "random":
            o = RawObservation.random_example(
                dim_state=dim_state,
                dim_action=dim_action,
                num_states=num_states,
                num_actions=num_actions,
            )
        else:
            with pytest.raises(ValueError):
                o = RawObservation.get_example(
                    dim_state=dim_state,
                    dim_action=dim_action,
                    num_states=num_states,
                    num_actions=num_actions,
                    kind=kind,
                )
            return

        if discrete:
            torch.testing.assert_allclose(o.state.shape, torch.Size([]))
            torch.testing.assert_allclose(o.action.shape, torch.Size([]))
            torch.testing.assert_allclose(o.next_state.shape, torch.Size([]))
            torch.testing.assert_allclose(o.next_action.shape, torch.Size([]))
            torch.testing.assert_allclose(o.log_prob_action, torch.tensor(1.0))

        else:
            torch.testing.assert_allclose(o.state.shape, torch.Size((dim_state,)))
            torch.testing.assert_allclose(o.action.shape, torch.Size((dim_action,)))
            torch.testing.assert_allclose(o.next_state.shape, torch.Size((dim_state,)))
            torch.testing.assert_allclose(
                o.next_action.shape, torch.Size((dim_action,))
            )
            torch.testing.assert_allclose(o.log_prob_action, torch.ones(dim_action))
