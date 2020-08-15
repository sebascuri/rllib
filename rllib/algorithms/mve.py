"""Model Based Value Expansion Algorithm."""
import torch

from rllib.util.neural_networks.utilities import repeat_along_dimension
from rllib.util.rollout import rollout_model
from rllib.util.value_estimation import mc_return
from rllib.value_function.integrate_q_value_function import IntegrateQValueFunction


def mve_expand(
    base_algorithm,
    dynamical_model,
    reward_model,
    num_steps=1,
    num_samples=15,
    termination=None,
):
    """Expand a MVE-Expanded algorithm.

    Given an algorithm, return the target via simulation of a model.
    """
    #

    class MVE(type(base_algorithm)):
        """Derived Algorithm using MVE to calculate targets.

        Overrides get_value_target() method.

        References
        ----------
        Feinberg, V., et. al. (2018).
        Model-based value estimation for efficient model-free reinforcement learning.
        arXiv.
        """

        def __init__(self, base_alg):
            super().__init__(**{**base_alg.__dict__, **dict(base_alg.named_modules())})
            self.dynamical_model = dynamical_model
            self.reward_model = reward_model
            self.num_steps = num_steps
            self.num_samples = num_samples
            self.termination = termination

            if not hasattr(self, "value_target") and hasattr(self, "q_target"):
                self.value_target = IntegrateQValueFunction(
                    self.q_target, self.policy, num_samples=self.num_samples
                )

        def get_value_target(self, observation):
            """Rollout model and call base algorithm with transitions."""
            if self.num_steps == 0:
                value = super().get_value_target(observation)
            else:
                with torch.no_grad():
                    state = repeat_along_dimension(
                        observation.state[..., 0, :], number=self.num_samples, dim=0
                    )
                    trajectory = rollout_model(
                        self.dynamical_model,
                        self.reward_model,
                        self.policy,
                        state,
                        max_steps=self.num_steps,
                        termination=self.termination,
                    )
                    value = mc_return(
                        trajectory,
                        gamma=self.gamma,
                        value_function=self.value_target,
                        reward_transformer=self.reward_transformer,
                    ).unsqueeze(-2)

            return value.mean(0)

    return MVE(base_algorithm)
