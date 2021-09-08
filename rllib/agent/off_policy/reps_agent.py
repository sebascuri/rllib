"""Implementation of REPS Agent."""

import torch
from torch.optim import Adam

from rllib.algorithms.reps import REPS
from rllib.policy import NNPolicy
from rllib.util.neural_networks.utilities import deep_copy_module
from rllib.value_function import NNValueFunction

from .off_policy_agent import OffPolicyAgent


class REPSAgent(OffPolicyAgent):
    """Implementation of the REPS algorithm.

    References
    ----------
    Peters, J., Mulling, K., & Altun, Y. (2010, July).
    Relative entropy policy search. AAAI.

    Deisenroth, M. P., Neumann, G., & Peters, J. (2013).
    A survey on policy search for robotics. Foundations and Trends® in Robotics.
    """

    def __init__(
        self,
        policy,
        critic,
        reps_eta=1.0,
        relent_regularization=False,
        learn_policy=True,
        num_iter=200,
        train_frequency=0,
        num_rollouts=15,
        reset_memory_after_learn=True,
        *args,
        **kwargs,
    ):
        super().__init__(
            num_iter=num_iter,
            train_frequency=train_frequency,
            num_rollouts=num_rollouts,
            reset_memory_after_learn=reset_memory_after_learn,
            *args,
            **kwargs,
        )

        self.algorithm = REPS(
            policy=policy,
            critic=critic,
            reps_eta=reps_eta,
            relent_regularization=relent_regularization,
            learn_policy=learn_policy,
            *args,
            **kwargs,
        )
        # Over-write optimizer.
        self.optimizer = type(self.optimizer)(
            [
                p
                for n, p in self.algorithm.named_parameters()
                if "target" not in n and "old_policy" not in n
            ],
            **self.optimizer.defaults,
        )

        self.policy = self.algorithm.policy

    def learn(self):
        """See `AbstractAgent.train_agent'."""
        old_policy = deep_copy_module(self.policy)
        self._optimize_dual()
        if hasattr(self.policy, "prior"):
            self.policy.prior = old_policy
            self.set_policy(self.policy)

        if self.algorithm.learn_policy:
            self._fit_policy()

        if self.reset_memory_after_learn:
            self.memory.reset()  # Erase memory.

    def _optimize_dual(self):
        """Optimize the dual function."""
        self._optimize_loss(loss_name="dual_loss")

    def _fit_policy(self):
        """Fit the policy optimizing the weighted negative log-likelihood."""
        self._optimize_loss(loss_name="policy_loss")

    def _optimize_loss(self, loss_name="dual_loss"):
        """Optimize the loss performing `num_iter' gradient steps."""
        #

        def closure():
            """Gradient calculation."""
            observation, idx, weight = self.memory.sample_batch(self.batch_size)

            self.optimizer.zero_grad()
            losses = self.algorithm(observation.clone())
            self.optimizer.zero_grad()
            loss = getattr(losses, loss_name)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.algorithm.parameters(), self.clip_gradient_val
            )

            return losses

        self._learn_steps(closure)

    @classmethod
    def default(cls, environment, critic=None, policy=None, lr=5e-3, *args, **kwargs):
        """See `AbstractAgent.default'."""
        if critic is None:
            critic = NNValueFunction.default(environment)
        if policy is None:
            policy = NNPolicy.default(environment)

        optimizer = Adam(critic.parameters(), lr=lr)

        return super().default(
            environment,
            policy=policy,
            critic=critic,
            optimizer=optimizer,
            *args,
            **kwargs,
        )
