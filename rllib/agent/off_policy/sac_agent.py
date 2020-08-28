"""Implementation of DQNAgent Algorithms."""
from itertools import chain

import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.sac import SoftActorCritic
from rllib.dataset.experience_replay import ExperienceReplay
from rllib.policy import NNPolicy
from rllib.value_function import NNEnsembleQFunction, NNQFunction

from .off_policy_agent import OffPolicyAgent


class SACAgent(OffPolicyAgent):
    """Implementation of a SAC agent.

    Parameters
    ----------
    q_function: AbstractQFunction
        q_function that is learned.
    criterion: nn.Module
        Criterion to minimize the TD-error.

    References
    ----------
    Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018).
    Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a
    stochastic actor. ICML.

    """

    def __init__(
        self, q_function, policy, criterion, eta, regularization=False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        q_function = NNEnsembleQFunction.from_q_function(
            q_function=q_function, num_heads=2
        )
        self.algorithm = SoftActorCritic(
            policy=policy,
            critic=q_function,
            criterion=criterion(reduction="none"),
            gamma=self.gamma,
            eta=eta,
            regularization=regularization,
        )

        self.optimizer = type(self.optimizer)(
            [
                p
                for name, p in self.algorithm.named_parameters()
                if "target" not in name
            ],
            **self.optimizer.defaults,
        )
        self.policy = self.algorithm.policy
        self.dist_params.update(tanh=True)

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See `AbstractAgent.default'."""
        q_function = NNQFunction.default(environment, non_linearity="ReLU")
        policy = NNPolicy.default(environment, non_linearity="ReLU")

        optimizer = Adam(chain(policy.parameters(), q_function.parameters()), lr=1e-3)
        criterion = loss.MSELoss
        memory = ExperienceReplay(max_len=50000, num_steps=0)

        return cls(
            q_function=q_function,
            policy=policy,
            criterion=criterion,
            optimizer=optimizer,
            memory=memory,
            eta=0.2,
            regularization=False,
            num_iter=50,
            train_frequency=50,
            batch_size=100,
            target_update_frequency=1,
            num_rollouts=0,
            comment=environment.name,
            *args,
            **kwargs,
        )
