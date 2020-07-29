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
        # if self.q_function.discrete_state:
        #     batch_size = get_batch_size(state.long())
        # else:
        #     batch_size = get_batch_size(state)
        batch_size = get_batch_size(state, self.dim_state)
        aux_size = (1,) if len(batch_size) == 0 else batch_size
        if len(aux_size) > 1:
            raise NotImplementedError
        else:
            aux_size = aux_size[0]

        # Epsilon part.
        probs = (
            self.epsilon / self.num_actions * torch.ones((aux_size, self.num_actions))
        )

        # Greedy part.
        a = torch.argmax(self.q_function(state), dim=-1)
        probs[torch.arange(aux_size), a] += 1 - self.epsilon

        if not batch_size:
            probs = probs.squeeze(0)
        return torch.log(probs)
