"""Abstract Algorithm."""

from abc import ABCMeta
from dataclasses import dataclass

import torch.jit
import torch.nn as nn

from rllib.dataset.utilities import stack_list_of_tuples
from rllib.util.neural_networks import deep_copy_module, update_parameters
from rllib.util.utilities import RewardTransformer
from rllib.value_function.abstract_value_function import (
    AbstractQFunction,
    AbstractValueFunction,
)
from rllib.value_function.nn_ensemble_value_function import NNEnsembleQFunction


class AbstractAlgorithm(nn.Module, metaclass=ABCMeta):
    """Abstract Algorithm template."""

    eps = 1e-12

    def __init__(
        self,
        gamma,
        policy,
        critic,
        criterion=nn.MSELoss,
        reward_transformer=RewardTransformer(),
        *args,
        **kwargs,
    ):
        super().__init__()
        self._info = {}
        self.gamma = gamma
        self.policy = policy
        self.policy_target = deep_copy_module(self.policy)
        self.critic = critic
        self.critic_target = deep_copy_module(self.critic)
        self.criterion = criterion
        self.reward_transformer = reward_transformer

    def get_value_target(self, observation):
        """Get Q target from the observation."""
        return observation.reward + self.gamma * self.critic(observation.state)

    def process_value_prediction(self, value_prediction, observation):
        """Return processed value prediction (e.g. clamped)."""
        return value_prediction

    def actor_loss(self, observation):
        """Get actor loss."""
        return Loss(loss=torch.tensor(0.0))

    def critic_loss(self, observation):
        """Get critic loss."""
        if self.critic is None:
            return Loss(loss=torch.tensor(0.0))

        if isinstance(self.critic, AbstractValueFunction):
            pred_q = self.critic(observation.state)
        elif isinstance(self.critic, AbstractQFunction):
            pred_q = self.critic(observation.state, observation.action)
        else:
            pred_q = torch.zeros_like(observation.reward)
        pred_q = self.process_value_prediction(pred_q, observation)

        with torch.no_grad():
            q_target = self.get_value_target(observation)
            if pred_q.shape != q_target.shape:
                assert isinstance(self.critic, NNEnsembleQFunction)
                q_target = q_target.unsqueeze(-1).repeat_interleave(
                    self.critic.num_heads, -1
                )

            td_error = pred_q - q_target

        critic_loss = self.criterion(pred_q, q_target)

        if isinstance(self.critic, NNEnsembleQFunction):
            critic_loss = critic_loss.sum(-1)
            td_error = td_error.sum(-1)

        return Loss(loss=critic_loss, critic_loss=critic_loss, td_error=td_error)

    def forward(self, observation):
        """Compute the loss."""
        if isinstance(observation, list) and len(observation) > 1:
            try:  # When possible, parallelize the trajectories.
                observation = [stack_list_of_tuples(observation)]
            except RuntimeError:
                pass
        return self.forward_slow(observation)

    def forward_slow(self, observation):
        """Compute the losses."""
        raise NotImplementedError

    @torch.jit.export
    def update(self):
        """Update algorithm parameters."""
        if self.critic is not None:
            update_parameters(self.critic_target, self.critic, tau=self.critic.tau)
        update_parameters(self.policy_target, self.policy, tau=self.policy.tau)

    @torch.jit.export
    def reset(self):
        """Reset algorithms parameters."""
        pass

    def info(self):
        """Return info parameters for logging."""
        return self._info


@dataclass
class Loss:
    """Basic Loss class.

    Other Parameters
    ----------------
    loss: Tensor.
        Combined loss to optimize.
    td_error: Tensor.
        TD-Error of critic.
    policy_loss: Tensor.
        Loss of policy optimization.
    regularization_loss: Tensor.
        Either KL-divergence or entropy bonus.
    dual_loss: Tensor.
        Loss of dual minimization problem.
    """

    loss: torch.Tensor
    td_error: torch.Tensor = torch.tensor(0.0)
    policy_loss: torch.Tensor = torch.tensor(0.0)
    critic_loss: torch.Tensor = torch.tensor(0.0)
    regularization_loss: torch.Tensor = torch.tensor(0.0)
    dual_loss: torch.Tensor = torch.tensor(0.0)
