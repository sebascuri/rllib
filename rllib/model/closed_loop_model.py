"""Closed Loop Model file."""

import torch

from rllib.util.utilities import tensor_to_distribution

from .transformed_model import TransformedModel


class ClosedLoopModel(TransformedModel):
    """Compute the next-state (or reward) for a model in closed loop with a policy.

    In general, the policy may not predict all the actions but rather a smaller number
    of them. Hence, the forward method accept an `action' parameter.

    From the model.dim_action, the first policy.dim_action come from the policy and the
    rest from the action in the `forward' method.
    If model.dim_action == policy.dim_action, then the action in the forward method is
    discarded.

    """

    def __init__(self, base_model, policy, *args, **kwargs):
        super().__init__(base_model=base_model, transformations=[], *args, **kwargs)
        self.base_model = base_model
        self.policy = policy

    def forward(self, state, action=None, next_state=None):
        """Compute the next state in closed loop."""
        pi = tensor_to_distribution(self.policy(state), **self.policy.dist_params)
        policy_actions = pi.rsample()

        if policy_actions.shape[-1] == self.dim_action[0]:
            return self.base_model(state, policy_actions)

        assert action is not None
        actions = torch.cat((policy_actions, action), dim=-1)

        return self.base_model(state, actions)
