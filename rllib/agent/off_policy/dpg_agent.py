"""Implementation of Deterministic Policy Gradient Algorithms."""
from itertools import chain

import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.dpg import DPG
from rllib.dataset.experience_replay import ExperienceReplay
from rllib.policy import NNPolicy
from rllib.util.parameter_decay import Constant, ParameterDecay
from rllib.value_function import NNQFunction

from .off_policy_agent import OffPolicyAgent


class DPGAgent(OffPolicyAgent):
    """Implementation of the Deterministic Policy Gradient Agent.

    The AbstractDDPGAgent algorithm implements the DPG-Learning algorithm except for
    the computation of the TD-Error, which leads to different algorithms.

    TODO: build compatible q-function approximation.
    TODO: Add policy update frequency.

    Parameters
    ----------
    q_function: AbstractQFunction
        q_function that is learned.
    policy: AbstractPolicy
        policy that is learned.
    criterion: nn.Module
    optimizer: nn.Optimizer
        q_function optimizer.
    memory: ExperienceReplay
        memory where to store the observations.

    References
    ----------
    Silver, David, et al. (2014) "Deterministic policy gradient algorithms." JMLR.
    Lillicrap et. al. (2016). CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING. ICLR.

    """

    def __init__(
        self,
        q_function,
        policy,
        criterion,
        optimizer,
        memory,
        exploration_noise,
        num_iter=1,
        batch_size=64,
        target_update_frequency=4,
        policy_noise=0.0,
        noise_clip=1.0,
        train_frequency=1,
        num_rollouts=0,
        policy_update_frequency=1,
        gamma=0.99,
        exploration_steps=0,
        exploration_episodes=0,
        tensorboard=False,
        comment="",
    ):
        super().__init__(
            optimizer=optimizer,
            memory=memory,
            batch_size=batch_size,
            num_iter=num_iter,
            target_update_frequency=target_update_frequency,
            train_frequency=train_frequency,
            num_rollouts=num_rollouts,
            policy_update_frequency=policy_update_frequency,
            gamma=gamma,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            tensorboard=tensorboard,
            comment=comment,
        )

        assert policy.deterministic, "Policy must be deterministic."
        self.algorithm = DPG(
            q_function,
            policy,
            criterion(reduction="none"),
            gamma,
            policy_noise,
            noise_clip,
        )
        self.policy = self.algorithm.policy

        if not isinstance(exploration_noise, ParameterDecay):
            exploration_noise = Constant(exploration_noise)

        self.params["exploration_noise"] = exploration_noise
        self.dist_params.update(
            add_noise=True, policy_noise=self.params["exploration_noise"]
        )

    def train(self, val=True):
        """Set the agent in training mode."""
        super().train(val)
        self.dist_params.update(
            add_noise=True, policy_noise=self.params["exploration_noise"]
        )

    def eval(self, val=True):
        """Set the agent in evaluation mode."""
        super().eval(val)
        self.dist_params.update(
            add_noise=False, policy_noise=self.params["exploration_noise"]
        )

    @classmethod
    def default(
        cls,
        environment,
        gamma=0.99,
        exploration_steps=0,
        exploration_episodes=0,
        tensorboard=False,
        test=False,
    ):
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
            deterministic=True,
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
            exploration_noise=0.1,
            policy_update_frequency=2,
            num_iter=1,
            batch_size=100,
            target_update_frequency=1,
            policy_noise=0.2,
            noise_clip=0.5,
            train_frequency=1,
            num_rollouts=0,
            gamma=gamma,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            tensorboard=tensorboard,
            comment=environment.name,
        )
