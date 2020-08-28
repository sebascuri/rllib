"""Epsilon Greedy Policy."""

import torch

from rllib.util.neural_networks import get_batch_size

from .abstract_q_function_policy import AbstractQFunctionPolicy


class EpsGreedy(AbstractQFunctionPolicy):
    """Implementation of Epsilon Greedy Policy.

    An epsilon greedy exploration strategy chooses the greedy strategy with probability
    1-epsilon, and a random action with probability epsilon.

    If eps_end and eps_decay are not set, then epsilon will be always eps_start.
    If not, epsilon will decay exponentially at rate eps_decay from eps_start to
    eps_end.

    """

    @property
    def epsilon(self):
        """Return epsilon."""
        return self.param()

    def forward(self, state):
        """See `AbstractQFunctionPolicy.forward'."""
        batch_size = get_batch_size(state, self.dim_state)
        aux_size = () if len(batch_size) == 0 else batch_size

        # Epsilon part.
        eps = torch.true_divide(self.epsilon, self.num_actions)
        probabilities = eps * torch.ones(*aux_size, self.num_actions)
        greedy = (1 - self.epsilon) * torch.ones(*aux_size, self.num_actions)

        # Greedy part.
        a = torch.argmax(self.q_function(state), dim=-1)
        probabilities.scatter_add_(dim=-1, index=a.unsqueeze(-1), src=greedy)

        if not batch_size:
            probabilities = probabilities.squeeze(0)
        return torch.log(probabilities)
