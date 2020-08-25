"""Abstract Algorithm."""

from abc import ABCMeta

import torch.jit
import torch.nn as nn

from rllib.dataset.datatypes import Loss, Observation
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.util.neural_networks import deep_copy_module, update_parameters
from rllib.util.utilities import RewardTransformer
from rllib.value_function.abstract_value_function import (
    AbstractQFunction,
    AbstractValueFunction,
)
from rllib.value_function.nn_ensemble_value_function import NNEnsembleQFunction


class AbstractAlgorithm(nn.Module, metaclass=ABCMeta):
    """Abstract Algorithm base class.

    Methods
    -------
    get_value_target(self, observation):
        Get the target to learn the critic.

    process_value_prediction(self, predicted_value, observation):
        Process a value prediction, usually returns the same predicted_value.

    actor_loss(self, observation) -> Loss:
        Return the loss of the actor.

    critic_loss(self, observation) -> Loss:
        Return the loss of the critic.

    forward_slow(self, observation) -> Loss:
        Compute the algorithm losses.

    update(self):
        Update the algorithm parameters. Useful for annealing.

    reset(self):
        Reset the optimization algorithm. Useful for copying a policy between iters.

    info(self):
        Get optimization info.

    Parameters
    ----------
    gamma: float.
        Discount factor of RL algorithm.
    policy: AbstractPolicy.
        Algorithm policy.
    critic: AbstractQFunction.
        Algorithm critic.
    criterion: Type[nn._Loss].
        Criterion to optimize the critic.
    reward_transformer: RewardTransformer.
        Transformations to optimize reward.
    """

    eps = 1e-12

    def __init__(
        self,
        gamma,
        policy,
        critic,
        criterion=nn.MSELoss(reduction="mean"),
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
        next_v = self.critic(observation.next_state)
        next_v = next_v * (1 - observation.done)
        return self.reward_transformer(observation.reward) + self.gamma * next_v

    def process_value_prediction(self, predicted_value, observation):
        """Return processed value prediction (e.g. clamped)."""
        return predicted_value

    def actor_loss(self, observation):
        """Get actor loss.

        This is different for each algorithm.

        Parameters
        ----------
        observation: Observation.
            Sampled observations.
            It is of shape B x N x d, where:
                - B is the batch size
                - N is the N-step return
                - d is the dimension of the attribute.

        Returns
        -------
        loss: Loss.
            Loss with parameters loss, policy_loss, and regularization_loss filled.
        """
        return Loss()

    def critic_loss(self, observation):
        """Get critic loss.

        This is usually computed using fitted value iteration and semi-gradients.
        critic_loss = criterion(pred_q, target_q.detach()).

        Parameters
        ----------
        observation: Observation.
            Sampled observations.
            It is of shape B x N x d, where:
                - B is the batch size
                - N is the N-step return
                - d is the dimension of the attribute.

        Returns
        -------
        loss: Loss.
            Loss with parameters loss, critic_loss, and td_error filled.
        """
        if self.critic is None:
            return Loss()

        if isinstance(self.critic, AbstractValueFunction):
            pred_q = self.critic(observation.state)
        elif isinstance(self.critic, AbstractQFunction):
            pred_q = self.critic(observation.state, observation.action)
        else:
            raise NotImplementedError
        pred_q = self.process_value_prediction(pred_q, observation)

        with torch.no_grad():  # Use semi-gradients.
            q_target = self.get_value_target(observation)
            if pred_q.shape != q_target.shape:  # Reshape in case of ensembles.
                assert isinstance(self.critic, NNEnsembleQFunction)
                q_target = q_target.unsqueeze(-1).repeat_interleave(
                    self.critic.num_heads, -1
                )

            td_error = pred_q - q_target  # no gradients for td-error.
            if self.criterion.reduction == "mean":
                td_error = torch.mean(td_error)
            elif self.criterion.reduction == "sum":
                td_error = torch.sum(td_error)

        critic_loss = self.criterion(pred_q, q_target)

        if isinstance(self.critic, NNEnsembleQFunction):
            # Ensembles have last dimension as ensemble head.
            critic_loss = critic_loss.sum(-1)
            td_error = td_error.sum(-1)

        return Loss(critic_loss=critic_loss, td_error=td_error)

    def forward(self, observation):
        """Compute the losses.

        Given an Observation, it will compute the losses.
        Given a list of Trajectories, it tries to stack them to vectorize operations.
        If it fails, will iterate over the trajectories.
        """
        if isinstance(observation, Observation):
            trajectories = [observation]
        elif len(observation) > 1:
            try:
                # When possible, stack to parallelize the trajectories.
                # This requires all trajectories to be equal of length.
                trajectories = [stack_list_of_tuples(observation)]
            except RuntimeError:
                trajectories = observation
        else:
            trajectories = observation

        self.reset_info(num_trajectories=len(trajectories))

        loss = Loss()
        for trajectory in trajectories:
            loss += self.actor_loss(trajectory)
            loss += self.critic_loss(trajectory)

        return loss / len(trajectories)

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

    @torch.jit.export
    def info(self):
        """Return info parameters for logging."""
        return self._info

    @torch.jit.export
    def reset_info(self, num_trajectories=1):
        """Reset info when the iteration starts."""
        self._info["num_trajectories"] = num_trajectories
