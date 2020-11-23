"""Stochastic Ensemble Value Expansion Algorithm."""
import torch

from rllib.dataset.datatypes import Loss
from rllib.model.utilities import PredictionStrategy
from rllib.util.multiprocessing import run_parallel_returns
from rllib.util.value_estimation import n_step_return
from rllib.value_function import NNEnsembleQFunction

from .mve import MVE


class STEVE(MVE):
    """Stochastic Ensemble Algorithm using STEVE to calculate targets.

    Overrides get_value_target() method.

    References
    ----------
    Buckman, J., Hafner, D., Tucker, G., Brevdo, E., & Lee, H. (2018).
    Sample-efficient reinforcement learning with stochastic ensemble value
    expansion. NeuRIPS.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            num_q = self.critic_target.num_heads
        except AttributeError:
            num_q = 1

        try:
            num_models = self.dynamical_model.base_model.num_heads
        except AttributeError:
            num_models = 1

        self.num_models = num_models
        self.num_q = num_q

    def model_augmented_critic_loss(self, observation):
        """Get Model-Based critic-loss."""
        pred_q = self.base_algorithm.get_value_prediction(observation)

        # Get target_q with semi-gradients.
        with torch.no_grad():
            target_q = self.get_value_target(observation)
            if pred_q.shape != target_q.shape:  # Reshape in case of ensembles.
                assert isinstance(self.critic, NNEnsembleQFunction)
                target_q = target_q.unsqueeze(-1).repeat_interleave(
                    self.critic.num_heads, -1
                )

        critic_loss = self.base_algorithm.criterion(pred_q, target_q)

        return Loss(critic_loss=critic_loss)

    def _get_member_target(self, model_idx, state, action, critic_target):
        """Get member target.

        Notes
        -----
        Helper method to paralelize the calculation through processes.
        """
        self.dynamical_model.set_head(model_idx)
        self.reward_model.set_head(model_idx)
        with torch.no_grad():
            observation = self.simulate(
                state, self.policy, initial_action=action, stack_obs=True
            )
            n_step_returns = n_step_return(
                observation,
                gamma=self.gamma,
                value_function=self.value_function,
                reward_transformer=self.reward_transformer,
                entropy_regularization=self.entropy_loss.eta.item(),
                reduction="none",
            )  # samples*batch x horizon x num_q
            value = n_step_returns.reshape(
                -1, 1, self.num_samples, self.num_steps, self.num_q
            )
        return value

    def get_value_target(self, observation):
        """Rollout model and call base algorithm with transitions."""
        critic_target = torch.zeros(
            observation.state.shape[: -len(self.dynamical_model.dim_state)]
            + (self.num_samples, self.num_steps + 1, self.num_models, self.num_q)
        )  # Critic target shape B x (H + 1) x M x Q
        td_return = n_step_return(
            observation,
            gamma=self.gamma,
            value_function=self.value_function,
            reward_transformer=self.reward_transformer,
            entropy_regularization=self.entropy_loss.eta.item(),
            reduction="none",
        )
        td_samples = td_return.unsqueeze(-2).repeat_interleave(self.num_samples, -2)
        td_model = td_samples.unsqueeze(-2).repeat_interleave(self.num_models, -2)
        if td_model.shape != critic_target[..., -1, :, :].shape:
            td_model = td_model.unsqueeze(1)
        critic_target[..., -1, :, :] = td_model

        with PredictionStrategy(
            self.dynamical_model, self.reward_model, prediction_strategy="set_head"
        ), torch.no_grad():
            state = observation.state[..., 0, :]
            action = observation.action[..., 0, :]
            value = run_parallel_returns(
                self._get_member_target,
                [(i, state, action, critic_target) for i in range(self.num_models)],
            )
            critic_target[..., :-1, :, :] = torch.stack(value, dim=4)

        mean_target = critic_target.mean(dim=(2, 4, 5))  # (samples, models, qs)
        weight_target = 1 / (self.eps + critic_target.var(dim=(2, 4, 5)))

        weights = weight_target / weight_target.sum(-1, keepdim=True)
        target_q = (weights * mean_target).sum(-1)
        return target_q
