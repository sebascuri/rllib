"""Implementation of SVG-0 Algorithm."""
from itertools import chain

import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.svg0 import SVG0
from rllib.dataset.experience_replay import ExperienceReplay
from rllib.policy import NNPolicy
from rllib.value_function import NNQFunction

from .off_policy_agent import OffPolicyAgent


class SVG0Agent(OffPolicyAgent):
    """Implementation of the SVG-0 Agent.

    Parameters
    ----------
    q_function: AbstractQFunction
        q_function that is learned.
    policy: AbstractPolicy
        policy that is learned.
    criterion: nn.Module

    References
    ----------
    Heess, N., Wayne, G., Silver, D., Lillicrap, T., Erez, T., & Tassa, Y. (2015).
    Learning continuous control policies by stochastic value gradients. NeuRIPS.

    """

    def __init__(self, q_function, policy, criterion, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert not policy.deterministic, "Policy must be stochastic."
        self.algorithm = SVG0(
            critic=q_function,
            policy=policy,
            criterion=criterion(reduction="none"),
            gamma=self.gamma,
        )
        self.policy = self.algorithm.policy

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See `AbstractAgent.default'."""
        q_function = NNQFunction(
            dim_state=environment.dim_state,
            dim_action=environment.dim_action,
            num_states=environment.num_states,
            num_actions=environment.num_actions,
            layers=[256, 256],
            biased_head=True,
            non_linearity="ReLU",
            tau=5e-3,
            input_transform=None,
        )
        policy = NNPolicy(
            dim_state=environment.dim_state,
            dim_action=environment.dim_action,
            num_states=environment.num_states,
            num_actions=environment.num_actions,
            action_scale=environment.action_scale,
            goal=environment.goal,
            layers=[256, 256],
            biased_head=True,
            non_linearity="ReLU",
            tau=5e-3,
            input_transform=None,
            deterministic=False,
        )

        optimizer = Adam(chain(policy.parameters(), q_function.parameters()), lr=3e-4)
        criterion = loss.MSELoss
        memory = ExperienceReplay(max_len=100000, num_steps=0)

        return cls(
            q_function=q_function,
            policy=policy,
            criterion=criterion,
            optimizer=optimizer,
            memory=memory,
            policy_update_frequency=2,
            num_iter=1,
            batch_size=100,
            target_update_frequency=1,
            train_frequency=1,
            num_rollouts=0,
            clip_gradient_val=10,
            comment=environment.name,
            *args,
            **kwargs,
        )
