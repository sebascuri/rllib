"""Model-Based BPTT Agent."""
from itertools import chain

import torch
import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.bptt import BPTT
from rllib.algorithms.model_learning_algorithm import ModelLearningAlgorithm
from rllib.model import EnsembleModel, TransformedModel
from rllib.policy import NNPolicy
from rllib.reward.quadratic_reward import QuadraticReward
from rllib.value_function import NNValueFunction

from .model_based_agent import ModelBasedAgent


class BPTTAgent(ModelBasedAgent):
    """Implementation of a Back-Propagation Through Time Agent."""

    def __init__(
        self,
        policy,
        critic,
        dynamical_model,
        reward_model,
        criterion,
        termination=None,
        num_steps=1,
        num_samples=15,
        algorithm=BPTT,
        *args,
        **kwargs,
    ):
        algorithm = algorithm(
            policy=policy,
            critic=critic,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            termination=termination,
            criterion=criterion(reduction="mean"),
            num_steps=num_steps,
            num_samples=num_samples,
            *args,
            **kwargs,
        )

        super().__init__(policy_learning_algorithm=algorithm, *args, **kwargs)

        self.optimizer = type(self.optimizer)(
            [
                p
                for name, p in self.algorithm.named_parameters()
                if ("model" not in name and "target" not in name and p.requires_grad)
            ],
            **self.optimizer.defaults,
        )

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See `AbstractAgent.default'."""
        test = kwargs.get("test", False)

        q_function = NNValueFunction(
            dim_state=environment.dim_state,
            num_states=environment.num_states,
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
        optimizer = Adam(chain(policy.parameters(), q_function.parameters()), lr=1e-3)
        criterion = loss.MSELoss

        model = EnsembleModel(
            dim_state=environment.dim_state,
            dim_action=environment.dim_action,
            num_heads=5,
            layers=[200, 200],
            biased_head=False,
            non_linearity="ReLU",
            input_transform=None,
            deterministic=False,
        )
        dynamical_model = TransformedModel(model, kwargs.get("transformations", list()))
        reward_model = kwargs.get(
            "reward_model",
            QuadraticReward(
                torch.eye(environment.dim_state[0]),
                torch.eye(environment.dim_action[0]),
                goal=environment.goal,
            ),
        )

        model_optimizer = Adam(dynamical_model.parameters(), lr=5e-4)

        model_learning_algorithm = ModelLearningAlgorithm(
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            num_epochs=2 if kwargs.get("test", False) else 30,
            batch_size=64,
            bootstrap=True,
            model_optimizer=model_optimizer,
        )

        return cls(
            policy=policy,
            critic=q_function,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            criterion=criterion,
            termination=None,
            num_steps=4,
            num_samples=15,
            learn_from_real=True,
            model_learning_algorithm=model_learning_algorithm,
            optimizer=optimizer,
            num_iter=5 if test else 50,
            batch_size=64,
            thompson_sampling=False,
            comment=environment.name,
            gamma=0.99,
            *args,
            **kwargs,
        )
