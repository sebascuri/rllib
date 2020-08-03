"""Implementation of Expected SARSA Agent."""

import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.esarsa import ESARSA
from rllib.policy import EpsGreedy
from rllib.util.parameter_decay import ExponentialDecay
from rllib.value_function import NNQFunction

from .on_policy_agent import OnPolicyAgent


class ExpectedSARSAAgent(OnPolicyAgent):
    """Implementation of an Expected SARSA agent.

    The SARSA agent implements the SARSA algorithm except for the
    computation of the TD-Error, which leads to different algorithms.

    Parameters
    ----------
    q_function: AbstractQFunction
        q_function that is learned.
    policy: QFunctionPolicy.
        Q-function derived policy.
    criterion: nn.Module
        Criterion to minimize the TD-error.
    batch_size: int
        Number of trajectory batches before performing a TD-pdate.
    optimizer: nn.optim
        Optimization algorithm for q_function.
    target_update_frequency: int
        How often to update the q_function target.
    gamma: float, optional
        Discount factor.
    exploration_steps: int, optional
        Number of random exploration steps.
    exploration_episodes: int, optional
        Number of random exploration steps.

    References
    ----------
    Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.
    """

    def __init__(self, q_function, policy, criterion, optimizer, *args, **kwargs):
        super().__init__(
            optimizer=optimizer,
            num_rollouts=kwargs.pop("num_rollouts", 0),
            train_frequency=kwargs.pop("train_frequency", 1),
            *args,
            **kwargs,
        )
        self.algorithm = ESARSA(
            q_function, criterion(reduction="mean"), policy, self.gamma
        )
        self.policy = policy

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See `AbstractAgent.default'."""
        q_function = NNQFunction(
            dim_state=environment.dim_state,
            dim_action=environment.dim_action,
            num_states=environment.num_states,
            num_actions=environment.num_actions,
            layers=[200, 200],
            biased_head=True,
            non_linearity="Tanh",
            tau=0,
            input_transform=None,
        )

        policy = EpsGreedy(q_function, ExponentialDecay(start=1.0, end=0.01, decay=500))
        optimizer = Adam(q_function.parameters(), lr=3e-4)
        criterion = loss.MSELoss

        return cls(
            q_function=q_function,
            policy=policy,
            optimizer=optimizer,
            criterion=criterion,
            num_iter=1,
            target_update_frequency=4,
            train_frequency=1,
            num_rollouts=0,
            comment=environment.name,
            *args,
            **kwargs,
        )
