"""Model Based Value Expansion Algorithm."""
import torch

from rllib.util.value_estimation import mb_return


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
        """Derived Algorithm using MVE to calculate targets."""

        def __init__(self, base_alg):
            super().__init__(**{**base_alg.__dict__, **dict(base_alg.named_modules())})
            self.dynamical_model = dynamical_model
            self.reward_model = reward_model
            self.num_steps = num_steps
            self.num_samples = num_samples
            self.termination = termination

        def get_q_target(self, observation):
            """Rollout model and call base algorithm with transitions."""
            with torch.no_grad():
                value, trajectory = mb_return(
                    observation.state,
                    self.dynamical_model,
                    self.reward_model,
                    self.policy,
                    num_steps=self.num_steps,
                    gamma=self.gamma,
                    value_function=self.value_function,
                    num_samples=self.num_samples,
                    entropy_reg=0.0,
                    termination=None,
                )

            return value

    return MVE(base_algorithm)
