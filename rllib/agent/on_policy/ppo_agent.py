"""Implementation of PPO Algorithm."""

import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.ppo import PPO
from rllib.policy import NNPolicy
from rllib.util.neural_networks.utilities import stop_learning
from rllib.value_function import NNValueFunction

from .on_policy_agent import OnPolicyAgent


class PPOAgent(OnPolicyAgent):
    """Implementation of the PPO Agent.

    References
    ----------
    Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
    Proximal policy optimization algorithms. ArXiv.
    """

    def __init__(
        self,
        policy,
        value_function,
        criterion,
        epsilon=0.2,
        lambda_=0.97,
        target_kl=0.01,
        weight_value_function=1.0,
        weight_entropy=0.01,
        monte_carlo_target=False,
        clamp_value=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.algorithm = PPO(
            critic=value_function,
            policy=policy,
            epsilon=epsilon,
            criterion=criterion(reduction="mean"),
            weight_value_function=weight_value_function,
            weight_entropy=weight_entropy,
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
    def default(cls, environment, *args, **kwargs):
        """See `AbstractAgent.default'."""
        policy = NNPolicy.default(environment)
        value_function = NNValueFunction.default(environment)

        optimizer = Adam(
            [
                {"params": policy.parameters(), "lr": 3e-4},
                {"params": value_function.parameters(), "lr": 1e-3},
            ]
        )
        criterion = loss.MSELoss

        return cls(
            policy=policy,
            value_function=value_function,
            optimizer=optimizer,
            criterion=criterion,
            epsilon=0.2,
            lambda_=0.95,
            target_kl=0.005,
            entropy_regularization=0.01,
            monte_carlo_target=False,
            clamp_value=True,
            num_iter=80,
            target_update_frequency=1,
            train_frequency=0,
            num_rollouts=4,
            comment=environment.name,
            *args,
            **kwargs,
        )
