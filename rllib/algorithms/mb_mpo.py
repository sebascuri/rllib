"""Model-Based implementation of Maximum a Posterior Policy Optimization algorithm."""

import torch

from rllib.dataset.utilities import stack_list_of_tuples
from rllib.util.value_estimation import mc_return

from .mpo import MPO
from .simulation_algorithm import SimulationAlgorithm


class MBMPO(MPO):
    """Model-Based implementation of Maximum a Posteriori Policy Optimizaiton."""

    def __init__(
        self,
        dynamical_model,
        reward_model,
        termination_model=None,
        num_particles=1,
        num_model_steps=1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.simulator = SimulationAlgorithm(
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            termination_model=termination_model,
            num_particles=num_particles,
            num_model_steps=num_model_steps,
        )

    def compute_mpo_loss(self, state, action):
        """Compute mpo loss for a given set of state/action pairs."""
        model_free_loss = super().compute_mpo_loss(state, action)
        with torch.no_grad():
            sim_trajectory = self.simulator.simulate(state, self.old_policy, action)
            sim_observation = stack_list_of_tuples(sim_trajectory, dim=-2)
            q_values = mc_return(
                sim_observation,
                gamma=self.gamma,
                td_lambda=self.td_lambda,
                value_function=self.value_target,
                reward_transformer=self.reward_transformer,
                entropy_regularization=self.entropy_loss.eta.detach().item(),
                reduction="none",
            )
        log_p, _ = self.get_log_p_and_ope_weight(
            sim_observation.state, sim_observation.action
        )

        model_based_mpo_loss = self.mpo_loss(
            q_values=q_values, action_log_p=log_p
        ).reduce(self.criterion.reduction)
        self._info.update(mpo_eta=self.mpo_loss.eta)
        return (model_free_loss + model_based_mpo_loss) / 2
