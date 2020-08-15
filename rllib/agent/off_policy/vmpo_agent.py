"""V-MPO Agent Implementation."""
from itertools import chain

import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.vmpo import VMPO
from rllib.dataset.experience_replay import ExperienceReplay
from rllib.policy import NNPolicy
from rllib.value_function import NNValueFunction

from .off_policy_agent import OffPolicyAgent


class VMPOAgent(OffPolicyAgent):
    """Implementation of an agent that runs V-MPO."""

    def __init__(
        self,
        policy,
        value_function,
        optimizer,
        memory,
        criterion,
        epsilon=0.1,
        epsilon_mean=0.1,
        epsilon_var=0.001,
        regularization=False,
        top_k_fraction=0.5,
        *args,
        **kwargs,
    ):
        super().__init__(memory=memory, optimizer=optimizer, *args, **kwargs)

        self.algorithm = VMPO(
            policy=policy,
            critic=value_function,
            criterion=criterion(reduction="none"),
            epsilon=epsilon,
            epsilon_mean=epsilon_mean,
            epsilon_var=epsilon_var,
            regularization=regularization,
            top_k_fraction=top_k_fraction,
            gamma=self.gamma,
        )

        self.policy = self.algorithm.policy
        self.optimizer = type(optimizer)(
            [
                p
                for name, p in self.algorithm.named_parameters()
                if "target" not in name
            ],
            **optimizer.defaults,
        )

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See `AbstractAgent.default'."""
        value_function = NNValueFunction(
            dim_state=environment.dim_state,
            num_states=environment.num_states,
            layers=[200, 200],
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
            layers=[100, 100],
            biased_head=True,
            non_linearity="ReLU",
            tau=5e-3,
            input_transform=None,
            deterministic=False,
        )

        optimizer = Adam(
            chain(policy.parameters(), value_function.parameters()), lr=5e-4
        )
        criterion = loss.MSELoss
        memory = ExperienceReplay(max_len=50000, num_steps=0)

        if environment.num_actions > 0:
            epsilon = 0.1
            epsilon_mean = 0.5
            epsilon_var = None
        else:
            epsilon = 0.1
            epsilon_mean = 0.1
            epsilon_var = 1e-4

        return cls(
            policy,
            value_function,
            optimizer,
            memory,
            criterion,
            epsilon=epsilon,
            epsilon_mean=epsilon_mean,
            epsilon_var=epsilon_var,
            regularization=False,
            top_k_fraction=0.5,
            num_iter=5 if kwargs.get("test", False) else 1000,
            batch_size=100,
            target_update_frequency=1,
            train_frequency=0,
            num_rollouts=2,
            comment=environment.name,
            *args,
            **kwargs,
        )

    def learn(self) -> None:
        """Learn with V-MPO (On-Policy?)."""
        super().learn()
        self.memory.reset()
