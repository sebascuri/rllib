"""Actor-Critic Algorithm."""
import torch

from rllib.dataset.utilities import stack_list_of_tuples
from rllib.util import discount_sum, integrate, tensor_to_distribution
from rllib.util.neural_networks import deep_copy_module, update_parameters

from .abstract_algorithm import AbstractAlgorithm, ACLoss


class ActorCritic(AbstractAlgorithm):
    r"""Implementation of Policy Gradient algorithm.

    Policy-Gradient is an on-policy model-free control algorithm.
    Policy-Gradient computes the policy gradient using a critic to estimate the returns
    (sum of discounted rewards).

    The Policy-Gradient algorithm is a policy gradient algorithm that estimates the
    gradient:
    .. math:: \grad J = \int_{\tau} \grad \log \pi(s_t) Q(s_t, a_t),
    where the previous integral is computed through samples (s_t, a_t) samples.


    Parameters
    ----------
    policy: AbstractPolicy
        Policy to optimize.
    critic: AbstractQFunction
        Critic that evaluates the current policy.
    criterion: _Loss
        Criterion to optimize the baseline.
    gamma: float
        Discount factor.

    References
    ----------
    Sutton, R. S., McAllester, D. A., Singh, S. P., & Mansour, Y. (2000).
    Policy gradient methods for reinforcement learning with function approximation.NIPS.

    Konda, V. R., & Tsitsiklis, J. N. (2000).
    Actor-critic algorithms. NIPS.
    """

    eps = 1e-12

    def __init__(self, policy, critic, criterion, gamma=0.99, num_samples=15):
        super().__init__()
        # Actor
        self.policy = policy
        self.policy_target = deep_copy_module(policy)

        # Critic
        self.critic = critic
        self.critic_target = deep_copy_module(critic)

        self.criterion = criterion
        self.gamma = gamma

        self.num_samples = num_samples

    def returns(self, trajectory):
        """Estimate the returns of a trajectory."""
        state, action = trajectory.state, trajectory.action
        pred_q = self.critic(state, action)
        returns = pred_q
        return returns

    def get_q_target(self, reward, next_state, done):
        """Get q function target."""
        next_pi = tensor_to_distribution(self.policy(next_state))
        next_v = integrate(
            lambda a: self.critic_target(next_state, a),
            next_pi,
            num_samples=self.num_samples,
        )
        return reward + self.gamma * next_v * (1 - done)

    def forward_slow(self, trajectories):
        """Compute the losses iterating through the trajectories."""
        actor_loss = torch.tensor(0.0)
        critic_loss = torch.tensor(0.0)
        td_error = torch.tensor(0.0)

        for trajectory in trajectories:
            state, action, reward, next_state, done, *r = trajectory

            # ACTOR LOSS
            pi = tensor_to_distribution(self.policy(state))
            if self.policy.discrete_action:
                action = action.long()
            with torch.no_grad():
                returns = self.returns(trajectory)
                returns = (returns - returns.mean()) / (returns.std() + self.eps)

            actor_loss += discount_sum(-pi.log_prob(action) * returns, self.gamma)

            # CRITIC LOSS
            with torch.no_grad():
                target_q = self.get_q_target(reward, next_state, done)

            pred_q = self.critic(state, action)
            critic_loss += self.criterion(pred_q, target_q).mean()
            td_error += (pred_q - target_q).detach().mean()

        num_trajectories = len(trajectories)
        return ACLoss(
            loss=(actor_loss + critic_loss) / num_trajectories,
            policy_loss=actor_loss / num_trajectories,
            critic_loss=critic_loss / num_trajectories,
            td_error=td_error / num_trajectories,
        )

    def forward(self, trajectories):
        """Compute the losses of a trajectory."""
        if len(trajectories) > 1:
            try:  # When possible, paralelize the trajectories.
                trajectories = [stack_list_of_tuples(trajectories)]
            except RuntimeError:
                pass
        return self.forward_slow(trajectories)

    def update(self):
        """Update the baseline network."""
        update_parameters(self.policy_target, self.policy, tau=self.policy.tau)
        update_parameters(self.critic_target, self.critic, tau=self.critic.tau)
