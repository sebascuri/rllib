"""Template for a Model Based Agent.

A model based agent has three behaviors:
- It learns models from data collected from the environment.
- It optimizes policies with simulated data from the models.
- It plans with the model and policies (as guiding sampler).
"""

from itertools import chain

import torch
from gym.utils import colorize
from torch.optim import Adam
from tqdm import tqdm

from rllib.agent.abstract_agent import AbstractAgent
from rllib.algorithms.model_learning_algorithm import ModelLearningAlgorithm
from rllib.algorithms.mpc.policy_shooting import PolicyShooting
from rllib.dataset.datatypes import Observation
from rllib.dataset.experience_replay import ExperienceReplay, StateExperienceReplay
from rllib.dataset.transforms import DeltaState, MeanFunction, StateNormalizer
from rllib.model import EnsembleModel, NNModel, TransformedModel
from rllib.policy.derived_policy import DerivedPolicy
from rllib.policy.mpc_policy import MPCPolicy
from rllib.util.neural_networks.utilities import DisableGradient
from rllib.util.utilities import tensor_to_distribution


class ModelBasedAgent(AbstractAgent):
    """Implementation of a Model Based RL Agent.

    Parameters
    ----------
    policy_learning_algorithm: PolicyLearningAlgorithm.
    model_learning_algorithm: ModelLearningAlgorithm
    planning_algorithm: MPCSolver.
    simulation_algorithm: SimulationAlgorithm.
    num_simulation_iterations: int.
    learn_from_real: bool.
        Flag that indicates whether or not to learn from real transitions.
    thompson_sampling: bool.
        Flag that indicates whether or not to use posterior sampling for the model.

    Other Parameters
    ----------------
    See AbstractAgent.
    """

    def __init__(
        self,
        num_rollouts=1,
        train_frequency=0,
        exploration_steps=0,
        exploration_episodes=1,
        model_learn_train_frequency=0,
        model_learn_num_rollouts=1,
        model_learn_exploration_steps=None,
        model_learn_exploration_episodes=None,
        policy_learning_algorithm=None,
        model_learning_algorithm=None,
        planning_algorithm=None,
        simulation_algorithm=None,
        num_simulation_iterations=0,
        learn_from_real=True,
        thompson_sampling=False,
        memory=None,
        *args,
        **kwargs,
    ):
        self.algorithm = policy_learning_algorithm
        super().__init__(
            num_rollouts=num_rollouts,
            train_frequency=train_frequency,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            *args,
            **kwargs,
        )
        self.model_learn_train_frequency = model_learn_train_frequency
        self.model_learn_num_rollouts = model_learn_num_rollouts

        if model_learn_exploration_steps is None:
            model_learn_exploration_steps = self.exploration_steps
        if model_learn_exploration_episodes is None:
            model_learn_exploration_episodes = self.exploration_episodes + 1
        self.model_learn_exploration_steps = model_learn_exploration_steps
        self.model_learn_exploration_episodes = model_learn_exploration_episodes

        self.planning_algorithm = planning_algorithm
        self.model_learning_algorithm = model_learning_algorithm
        self.simulation_algorithm = simulation_algorithm

        if policy_learning_algorithm is not None:
            policy = policy_learning_algorithm.policy
        elif planning_algorithm is not None:
            policy = MPCPolicy(self.planning_algorithm)
        else:
            raise NotImplementedError

        dynamical_model, reward_model, termination_model = None, None, None
        for alg in [
            policy_learning_algorithm,
            planning_algorithm,
            model_learning_algorithm,
            simulation_algorithm,
        ]:
            if alg is not None:
                dynamical_model, reward_model = alg.dynamical_model, alg.reward_model
                termination_model = alg.termination_model
                break

        self.dynamical_model = dynamical_model
        self.reward_model = reward_model
        self.termination_model = termination_model
        assert self.dynamical_model.model_kind == "dynamics"
        assert self.reward_model.model_kind == "rewards"
        if self.termination_model is not None:
            assert self.termination_model.model_kind == "termination"

        self.policy = DerivedPolicy(policy, self.dynamical_model.base_model.dim_action)
        self.num_simulation_iterations = num_simulation_iterations
        self.learn_from_real = learn_from_real
        self.learn_from_sim = self.num_simulation_iterations > 0
        self.thompson_sampling = thompson_sampling

        if self.thompson_sampling:
            self.dynamical_model.set_prediction_strategy("posterior")

        if memory is None:
            memory = ExperienceReplay(max_len=100000, num_steps=0)
        self.memory = memory
        self.initial_states_dataset = StateExperienceReplay(
            max_len=1000, dim_state=self.dynamical_model.dim_state
        )

    def act(self, state):
        """Ask the agent for an action to interact with the environment.

        If the plan horizon is zero, then it just samples an action from the policy.
        If the plan horizon > 0, then is plans with the current model.
        """
        if isinstance(self.planning_algorithm, PolicyShooting):
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.get_default_dtype())
            policy = tensor_to_distribution(
                self.policy(state), **self.policy.dist_params
            )
            self.pi = policy
            action = self.planning_algorithm(state).detach().numpy()
        else:
            action = super().act(state)

        dim = self.dynamical_model.base_model.dim_action[0]
        action = action[..., :dim]
        return action.clip(
            -self.policy.action_scale.numpy(), self.policy.action_scale.numpy()
        )

    def observe(self, observation):
        """Observe a new transition.

        If the episode is new, add the initial state to the state transitions.
        Add the transition to the data set.
        """
        super().observe(observation)
        if self.training:
            self.memory.append(observation)
        if self.learn_model_at_observe:
            self.model_learning_algorithm.learn(self.logger)
        if self.train_at_observe and len(self.memory) > self.batch_size:
            self.learn()

    def start_episode(self):
        """See `AbstractAgent.start_episode'."""
        super().start_episode()

        if self.thompson_sampling:
            self.dynamical_model.sample_posterior()

    def end_episode(self):
        """See `AbstractAgent.end_episode'.

        If the agent is training, and the base model is a GP Model, then add the
        transitions to the GP, and summarize and sparsify the GP Model.

        Then train the agent.
        """
        self.initial_states_dataset.append(self.last_trajectory[0].state.unsqueeze(0))
        if self.model_learning_algorithm is not None and self.training:
            self.model_learning_algorithm.add_last_trajectory(self.last_trajectory)
        if self.learn_model_at_end_episode:
            self.model_learning_algorithm.learn(self.logger)
        if self.train_at_end_episode:
            self.learn()
        super().end_episode()

    def learn(self):
        """Learn a policy with the model."""
        if self.learn_from_sim:
            print(colorize("Optimizing Policy with Simulated Data", "yellow"))
            self.simulate_and_learn_policy()

        if self.learn_from_real:
            print(colorize("Optimizing Policy with Real Data", "yellow"))
            self.learn_policy_from_real_data()

    def simulate_and_learn_policy(self):
        """Simulate the model and optimize the policy with the learned data.

        This consists of two steps:
            Step 1: Simulate trajectories with the model.
            Step 2: Implement a model free RL method that optimizes the policy.
        """
        self.dynamical_model.eval()
        # self.simulation_algorithm.dataset.reset()

        with DisableGradient(
            self.dynamical_model, self.reward_model, self.termination_model
        ):
            for _ in tqdm(range(self.num_simulation_iterations)):
                # Step 1: Simulate the state distribution
                with torch.no_grad():
                    self.policy.reset()  # TODO: Add goal distribution.
                    initial_states = self.simulation_algorithm.get_initial_states(
                        self.initial_states_dataset, self.memory
                    )

                    self.simulation_algorithm.simulate(
                        initial_states, self.policy, logger=self.logger
                    )

                # Step 2: Optimize policy with simulated data.
                self.learn_policy_from_sim_data()

    def learn_policy_from_sim_data(self):
        """Learn policy using simulated transitions."""
        #

        def closure():
            """Gradient calculation."""
            state = self.simulation_algorithm.dataset.sample_batch(self.batch_size)
            observation = Observation(state=state.unsqueeze(-2))
            self.optimizer.zero_grad()
            losses = self.algorithm(observation)
            losses.combined_loss.mean().backward()

            torch.nn.utils.clip_grad_norm_(
                self.algorithm.parameters(), self.clip_gradient_val
            )

            return losses

        self._learn_steps(closure)

    def learn_policy_from_real_data(self):
        """Learn policy using real transitions and use the model to predict targets."""
        #

        def closure():
            """Gradient calculation."""
            observation, *_ = self.memory.sample_batch(self.batch_size)
            self.optimizer.zero_grad()
            losses = self.algorithm(observation)
            losses.combined_loss.mean().backward()

            torch.nn.utils.clip_grad_norm_(
                self.algorithm.parameters(), self.clip_gradient_val
            )
            return losses

        self._learn_steps(closure)

    @property
    def learn_model_at_observe(self):
        """Raise flag if learn the model after observe."""
        return (
            self.training
            and self.model_learning_algorithm is not None
            and self.total_steps >= self.model_learn_exploration_steps
            and self.total_episodes >= self.model_learn_exploration_episodes
            and self.model_learn_train_frequency > 0
            and self.total_steps % self.model_learn_train_frequency == 0
        )

    @property
    def learn_model_at_end_episode(self):
        """Raise flag to learn the model at end of an episode."""
        return (
            self.training
            and self.model_learning_algorithm is not None
            and self.total_steps >= self.model_learn_exploration_steps
            and self.total_episodes >= self.model_learn_exploration_episodes
            and self.model_learn_num_rollouts > 0
            and (self.total_episodes + 1) % self.model_learn_num_rollouts == 0
        )

    @classmethod
    def default(
        cls, environment, dynamical_model=None, reward_model=None, *args, **kwargs
    ):
        """Get a default model-based agent."""
        if dynamical_model is None:
            model = EnsembleModel.default(environment, deterministic=False)
            dynamical_model = TransformedModel(
                model, [StateNormalizer(), MeanFunction(DeltaState())]
            )

        if reward_model is None:
            reward_model = TransformedModel(
                NNModel.default(environment, model_kind="rewards", deterministic=False),
                dynamical_model.forward_transformations,
            )

        params = list(chain(dynamical_model.parameters(), reward_model.parameters()))
        if len(params):
            model_optimizer = Adam(params, lr=1e-3, weight_decay=1e-4)

            model_learning_algorithm = ModelLearningAlgorithm(
                dynamical_model=dynamical_model,
                reward_model=reward_model,
                num_epochs=4 if kwargs.get("test", False) else 50,
                model_optimizer=model_optimizer,
            )
        else:
            model_learning_algorithm = None

        return super().default(
            environment,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            model_learning_algorithm=model_learning_algorithm,
            *args,
            **kwargs,
        )
