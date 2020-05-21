"""Implementation of a Transformation that clamps the next step attributes."""

import torch.jit
from .abstract_transform import AbstractTransform
from rllib.dataset.datatypes import Observation


class NextStateClamper(AbstractTransform):
    """Implementation of a Next Scale Clamper.

    Given a next state, it will saturate it in the inverse transformation.

    This helps to limit the forward propagation of models.

    Parameters
    ----------
    lower: torch.tensor
    higher: torch.tensor
    """

    def __init__(self, lower, higher):
        super().__init__()
        self.lower = lower
        self.higher = higher

    def forward(self, observation: Observation):
        """See `AbstractTransform.__call__'."""
        return observation

    @torch.jit.export
    def inverse(self, observation: Observation):
        """See `AbstractTransform.inverse'."""
        return Observation(
            state=observation.state,
            action=observation.action,
            reward=observation.reward,
            next_state=torch.max(torch.min(observation.next_state, self.higher),
                                 -self.lower),
            done=observation.done,
            next_action=observation.next_action,
            log_prob_action=observation.log_prob_action,
            entropy=observation.entropy,
            state_scale_tril=observation.state_scale_tril,
            next_state_scale_tril=observation.next_state_scale_tril
        )
