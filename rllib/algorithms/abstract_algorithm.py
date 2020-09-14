"""Abstract Algorithm."""

from abc import ABCMeta

import torch.jit
import torch.nn as nn

from rllib.dataset.datatypes import Loss, Observation
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.util.neural_networks import (
    deep_copy_module,
    freeze_parameters,
    update_parameters,
)
from rllib.util.utilities import (
    RewardTransformer,
    get_entropy_and_log_p,
    off_policy_weight,
    separated_kl,
    tensor_to_distribution,
)
from rllib.value_function import (
    AbstractQFunction,
    AbstractValueFunction,
    IntegrateQValueFunction,
)
from rllib.value_function.nn_ensemble_value_function import NNEnsembleQFunction

from .kl_loss import KLLoss


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
        entropy_regularization=0.0,
        epsilon_mean=0.0,
        epsilon_var=0.0,
        regularization=True,
        num_samples=1,
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
        self.entropy_regularization = entropy_regularization
        self.kl_loss = KLLoss(
            epsilon_mean=epsilon_mean,
            epsilon_var=epsilon_var,
            regularization=regularization,
        )
        self.num_samples = num_samples
        self.post_init()

    def post_init(self):
        """Set derived modules after initialization."""
        self.value_target = IntegrateQValueFunction(
            self.critic_target, self.policy, num_samples=self.num_samples
        )
        if self.policy is not None:
            old_policy = deep_copy_module(self.policy)
            freeze_parameters(old_policy)
            self.old_policy = old_policy

    def set_policy(self, new_policy):
        """Set new policy."""
        self.policy = new_policy
        self.policy_target = deep_copy_module(self.policy)
        self.post_init()

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

        # Get pred_q at the current state, action pairs.
        if isinstance(self.critic, AbstractValueFunction):
            pred_q = self.critic(observation.state)
        elif isinstance(self.critic, AbstractQFunction):
            pred_q = self.critic(observation.state, observation.action)
        else:
            raise NotImplementedError
        pred_q = self.process_value_prediction(pred_q, observation)

        # Get target_q with semi-gradients.
        with torch.no_grad():
            target_q = self.get_value_target(observation)
            if pred_q.shape != target_q.shape:  # Reshape in case of ensembles.
                assert isinstance(self.critic, NNEnsembleQFunction)
                target_q = target_q.unsqueeze(-1).repeat_interleave(
                    self.critic.num_heads, -1
                )

            td_error = pred_q - target_q  # no gradients for td-error.
            if self.criterion.reduction == "mean":
                td_error = torch.mean(td_error)
            elif self.criterion.reduction == "sum":
                td_error = torch.sum(td_error)

        critic_loss = self.criterion(pred_q, target_q)

        if isinstance(self.critic, NNEnsembleQFunction):
            # Ensembles have last dimension as ensemble head; sum all ensembles.
            critic_loss = critic_loss.sum(-1)
            td_error = td_error.sum(-1)

        # Take mean over time coordinate.
        critic_loss = critic_loss.mean(-1)
        td_error = td_error.mean(-1)

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
            loss += self.regularization_loss(trajectory)

        return loss / len(trajectories)

    def get_log_p_kl_entropy(self, state, action):
        """Get kl divergence and current policy at a given state.

        Compute the separated KL divergence between current and old policy.
        When the policy is a MultivariateNormal distribution, it compute the divergence
        that correspond to the mean and the covariance separately.

        When the policy is a Categorical distribution, it computes the divergence and
        assigns it to the mean component. The variance component is kept to zero.

        Parameters
        ----------
        state: torch.Tensor
            Empirical state distribution.

        action: torch.Tensor
            Actions sampled by pi_old.

        Returns
        -------
        log_p: torch.Tensor
            Log probability of actions according to current policy.
        log_p_old: torch.Tensor
            Log probability of actions according to old policy.
        kl_mean: torch.Tensor
            KL-Divergence due to the change in the mean between current and
            previous policy.
        kl_var: torch.Tensor
            KL-Divergence due to the change in the variance between current and
            previous policy.
        """
        pi = tensor_to_distribution(self.policy(state), **self.policy.dist_params)
        pi_old = tensor_to_distribution(
            self.old_policy(state), **self.policy.dist_params
        )

        entropy, log_p = get_entropy_and_log_p(pi, action, self.policy.action_scale)
        _, log_p_old = get_entropy_and_log_p(pi_old, action, self.policy.action_scale)

        if isinstance(pi, torch.distributions.MultivariateNormal):
            kl_mean, kl_var = separated_kl(p=pi_old, q=pi)
        else:
            try:
                kl_mean = torch.distributions.kl_divergence(p=pi_old, q=pi).mean()
            except NotImplementedError:
                kl_mean = (log_p_old - log_p).mean()  # Approximate the KL with samples.
            kl_var = torch.zeros_like(kl_mean)

        num_t = self._info["num_trajectories"]
        self._info.update(
            kl_div=self._info["kl_div"] + (kl_mean + kl_var) / num_t,
            kl_mean=self._info["kl_mean"] + kl_mean / num_t,
            kl_var=self._info["kl_var"] + kl_var / num_t,
            entropy=self._info["entropy"] + entropy / num_t,
        )

        return log_p, log_p_old, kl_mean, kl_var, entropy

    def get_ope_weight(self, state, action, log_prob_action):
        """Get off-policy weight of a given transition."""
        pi = tensor_to_distribution(self.policy(state), **self.policy.dist_params)
        _, log_p = get_entropy_and_log_p(pi, action, self.policy.action_scale)

        weight = off_policy_weight(log_p, log_prob_action, full_trajectory=False)
        return weight

    def regularization_loss(self, observation):
        """Compute regularization loss."""
        _, _, kl_mean, kl_var, entropy = self.get_log_p_kl_entropy(
            observation.state, observation.action
        )
        entropy_loss = Loss(regularization_loss=-self.entropy_regularization * entropy)
        kl_loss = self.kl_loss(kl_mean, kl_var)
        self._info.update(
            eta_mean=self.kl_loss.eta_mean(), eta_var=self.kl_loss.eta_var()
        )
        return entropy_loss + kl_loss

    @torch.jit.export
    def update(self):
        """Update algorithm parameters."""
        if self.critic is not None:
            update_parameters(self.critic_target, self.critic, tau=self.critic.tau)
        update_parameters(self.policy_target, self.policy, tau=self.policy.tau)

    @torch.jit.export
    def reset(self):
        """Reset the optimization (kl divergence) for the next epoch."""
        # Copy over old policy for KL divergence
        self.old_policy.load_state_dict(self.policy.state_dict())

    @torch.jit.export
    def info(self):
        """Return info parameters for logging."""
        return self._info

    @torch.jit.export
    def reset_info(self, num_trajectories=1):
        """Reset info when the iteration starts."""
        self._info.update(
            num_trajectories=num_trajectories,
            kl_div=torch.tensor(0.0),
            kl_mean=torch.tensor(0.0),
            kl_var=torch.tensor(0.0),
            entropy=torch.tensor(0.0),
        )
