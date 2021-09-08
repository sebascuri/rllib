"""Python Script Template."""

import torch

from rllib.policy import TabularPolicy


class TestTabularPolicy(object):
    def test_init(self):
        policy = TabularPolicy(num_states=4, num_actions=2)
        torch.testing.assert_allclose(policy.table, torch.ones(2, 4))

    def test_set_value(self):
        policy = TabularPolicy(num_states=4, num_actions=2)
        policy.set_value(2, torch.tensor(1))
        l1 = torch.log(torch.tensor(1e-12))
        l2 = torch.log(torch.tensor(1.0 + 1e-12))
        torch.testing.assert_allclose(
            policy.table, torch.tensor([[1.0, 1.0, l1, 1], [1.0, 1.0, l2, 1]])
        )

        policy.set_value(0, torch.tensor([0.3, 0.7]))
        torch.testing.assert_allclose(
            policy.table, torch.tensor([[0.3, 1.0, l1, 1], [0.7, 1.0, l2, 1]])
        )
