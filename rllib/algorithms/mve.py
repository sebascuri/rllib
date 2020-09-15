"""Model Based Value Expansion Algorithm."""
import torch

from rllib.util.value_estimation import discount_cumsum, mc_return

from .abstract_mb_algorithm import AbstractMBAlgorithm


def mve_expand(
    base_algorithm,
    dynamical_model,
    reward_model,
    num_steps=1,
    num_samples=15,
    termination_model=None,
    td_k=False,
    *args,
    **kwargs,
):
    """Expand a MVE-Expanded algorithm.

    Given an algorithm, return the target via simulation of a model.
    """
    #

    class MVE(type(base_algorithm), AbstractMBAlgorithm):
        """Derived Algorithm using MVE to calculate targets.

        References
        ----------
        Feinberg, V., et. al. (2018).
        Model-based value estimation for efficient model-free reinforcement learning.
        arXiv.
        """

        def __init__(self):
            super().__init__(
                **{**base_algorithm.__dict__, **dict(base_algorithm.named_modules())}
            )
            AbstractMBAlgorithm.__init__(
                self,
                dynamical_model,
                reward_model,
                num_steps=num_steps,
                num_samples=num_samples,
                termination_model=termination_model,
            )

            self.base_algorithm_name = base_algorithm.__class__.__name__

            self.td_k = td_k
            self.policy.dist_params.update(**base_algorithm.policy.dist_params)
            self.policy_target.dist_params.update(
                **base_algorithm.policy_target.dist_params
            )
            self.entropy_loss = base_algorithm.entropy_loss
            self.kl_loss = base_algorithm.kl_loss
            self.ope = None

        def critic_loss(self, observation):
            """Get critic-loss by rolling out a model."""
            with torch.no_grad():
                state = observation.state[..., 0, :]
                observation = self.simulate(state, self.policy)
            if not self.td_k:
                observation.state = observation.state[..., :1, :]
                observation.action = observation.action[..., :1, :]

            return super().critic_loss(observation)

        def get_value_target(self, observation):
            """Rollout model and call base algorithm with transitions."""
            if self.td_k:
                final_state = observation.next_state[..., -1, :]
                done = observation.done[..., -1]
                final_value = self.value_target(final_state)

                if final_value.ndim == observation.reward.ndim:  # It is an ensemble.
                    final_value = final_value.min(-1)[0]

                rewards = torch.cat(
                    (observation.reward, (final_value * (1 - done)).unsqueeze(-1)),
                    dim=-1,
                )
                sim_target = discount_cumsum(
                    rewards, self.gamma, self.reward_transformer
                )[..., :-1]
            else:
                sim_target = mc_return(
                    observation,
                    gamma=self.gamma,
                    value_function=self.value_target,
                    reward_transformer=self.reward_transformer,
                    reduction="min",
                ).unsqueeze(-1)

            return sim_target

    return MVE()
