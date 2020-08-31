"""Implementation of PPO Algorithm."""

from rllib.algorithms.ppo import PPO
from rllib.util.neural_networks.utilities import stop_learning
from rllib.value_function import NNValueFunction

from .actor_critic_agent import ActorCriticAgent


class PPOAgent(ActorCriticAgent):
    """Implementation of the PPO Agent.

    References
    ----------
    Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
    Proximal policy optimization algorithms. ArXiv.
    """

    def __init__(
        self,
        policy,
        critic,
        epsilon=0.2,
        lambda_=0.95,
        target_kl=0.005,
        entropy_regularization=0.01,
        monte_carlo_target=False,
        clamp_value=True,
        *args,
        **kwargs,
    ):
        super().__init__(critic=critic, policy=policy, *args, **kwargs)
        self.algorithm = PPO(
            critic=critic,
            policy=policy,
            epsilon=epsilon,
            criterion=self.algorithm.criterion,
            entropy_regularization=entropy_regularization,
            monte_carlo_target=monte_carlo_target,
            clamp_value=clamp_value,
            lambda_=lambda_,
            gamma=self.gamma,
        )

        self.policy = self.algorithm.policy
        self.target_kl = target_kl

    def early_stop(self, *args, **kwargs):
        """Early stop the training algorithm."""
        if (
            kwargs.get("kl_div", kwargs.get("approx_kl_div", self.target_kl))
            >= 1.5 * self.target_kl
        ):
            stop_learning(self.policy)
        return False

    @classmethod
    def default(cls, environment, num_iter=80, *args, **kwargs):
        """See `AbstractAgent.default'."""
        return super().default(
            environment,
            critic=NNValueFunction.default(environment),
            num_iter=num_iter,
            *args,
            **kwargs,
        )
