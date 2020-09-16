"""Stochastic Ensemble Value Expansion Algorithm."""
import torch

from rllib.dataset.datatypes import Loss
from rllib.model.utilities import PredictionStrategy
from rllib.util.training.utilities import sharpness
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

    def get_value_target(self, observation):
        """Rollout model and call base algorithm with transitions."""
        critic_target = torch.zeros(
            observation.state.shape[: -len(self.dynamical_model.dim_state)]
            + (self.num_samples, self.num_steps, self.num_models, self.num_q)
        )  # Critic target shape B x (H + 1) x M x Q

        real_target_q = super().get_value_target(observation)  # TD-Target B x 1.

        with PredictionStrategy(
            self.dynamical_model, self.reward_model, prediction_strategy="set_head"
        ), torch.no_grad():
            state = observation.state[..., 0, :]
            action = observation.action[..., 0, :]
            for model_idx in range(self.num_models):  # Rollout each model.
                self.dynamical_model.set_head(model_idx)
                self.reward_model.set_head(model_idx)
                observation = self.simulate(state, self.policy, initial_action=action)
                fast_value = n_step_return(
                    observation,
                    gamma=self.gamma,
                    value_function=self.value_target,
                    reward_transformer=self.reward_transformer,
                    entropy_regularization=self.entropy_loss.eta.item(),
                    reduction="none",
                )  # samples*batch x horizon x num_q
                fv = fast_value.reshape(
                    -1, self.num_samples, self.num_steps, self.num_q
                )
                critic_target[..., model_idx, :] = fv.unsqueeze(1)

        mean_target = critic_target.mean(dim=(2, 4, 5))  # (samples, models, qs)
        weight_target = 1 / (self.eps + critic_target.var(dim=(2, 4, 5)))

        weights = weight_target / weight_target.sum(-1, keepdim=True)
        model_target_q = (weights * mean_target).sum(-1)

        sharpness_ = sharpness(self.dynamical_model, observation) + sharpness(
            self.reward_model, observation
        )
        alpha = 1.0 / (1.0 + sharpness_)
        target_q = alpha * model_target_q + (1 - alpha) * real_target_q

        return target_q
