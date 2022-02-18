"""Template for a Model Based Agent.

A model based agent has three behaviors:
- It learns models from data collected from the environment.
- It optimizes policies with simulated data from the models.
- It plans with the model and policies (as guiding sampler).
"""

from itertools import chain

import torch
from torch.optim import Adam

from rllib.agent.abstract_agent import AbstractAgent
from rllib.algorithms.model_learning_algorithm import ModelLearningAlgorithm
from rllib.dataset.experience_replay import ExperienceReplay, StateExperienceReplay
from rllib.environment.fake_environment import FakeEnvironment
from rllib.model import TransformedModel
from rllib.policy.random_policy import RandomPolicy
from rllib.util.neural_networks.utilities import DisableGradient
from rllib.util.rollout import rollout_policy


class ModelBasedAgent(AbstractAgent):
    """Implementation of a Model Based RL Agent.

    Parameters
    ----------
    policy_learning_algorithm: PolicyLearningAlgorithm.
    model_learning_algorithm: ModelLearningAlgorithm
    thompson_sampling: bool.
        Flag that indicates whether or not to use posterior sampling for the model.

    Other Parameters
    ----------------
    See AbstractAgent.
    """

    def __init__(
        self,
        dynamical_model,
        reward_model,
        termination_model=None,
        num_rollouts=0,
        train_frequency=50,
        num_iter=50,
        exploration_steps=0,
        exploration_episodes=1,
        model_learn_train_frequency=0,
        model_learn_num_rollouts=1,
        model_learn_exploration_steps=None,
        model_learn_exploration_episodes=None,
        policy_learning_algorithm=None,
        model_learning_algorithm=None,
        thompson_sampling=False,
        policy=None,
        memory=None,
        batch_size=100,
        clip_grad_val=10.0,
        simulation_frequency=1,
        simulation_max_steps=1000,
        num_memory_samples=0,
        num_initial_state_samples=1,
        num_initial_distribution_samples=0,
        initial_distribution=None,
        augment_dataset_with_sim=False,
        pre_train_iterations=0,
        *args,
        **kwargs,
    ):
        self.algorithm = policy_learning_algorithm
        super().__init__(
            num_rollouts=num_rollouts,
            train_frequency=train_frequency,
            num_iter=num_iter,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            clip_grad_val=clip_grad_val,
            batch_size=batch_size,
            *args,
            **kwargs,
        )
        self.model_learn_train_frequency = model_learn_train_frequency
        self.model_learn_num_rollouts = model_learn_num_rollouts

        if model_learn_exploration_steps is None:
            model_learn_exploration_steps = self.exploration_steps
        if model_learn_exploration_episodes is None:
            model_learn_exploration_episodes = self.exploration_episodes - 1
        self.model_learn_exploration_steps = model_learn_exploration_steps
        self.model_learn_exploration_episodes = model_learn_exploration_episodes

        self.model_learning_algorithm = model_learning_algorithm

        self.dynamical_model = dynamical_model
        self.reward_model = reward_model
        self.termination_model = termination_model
        assert self.dynamical_model.model_kind == "dynamics"
        assert self.reward_model.model_kind == "rewards"
        if self.termination_model is not None:
            assert self.termination_model.model_kind == "termination"

        if policy is None:
            if policy_learning_algorithm:
                policy = policy_learning_algorithm.policy
            else:
                RandomPolicy(dynamical_model.dim_state, dynamical_model.dim_action)

        self.policy = policy
        self.thompson_sampling = thompson_sampling

        if self.thompson_sampling:
            self.dynamical_model.set_prediction_strategy("posterior")

        if memory is None:
            memory = ExperienceReplay(max_len=100000, num_memory_steps=0)
        self.memory = memory
        self.initial_states_dataset = StateExperienceReplay(
            max_len=1000, dim_state=self.dynamical_model.dim_state
        )

        self.simulation_frequency = simulation_frequency
        self.simulation_max_steps = simulation_max_steps
        self.num_memory_samples = num_memory_samples
        self.num_initial_state_samples = num_initial_state_samples
        self.num_initial_distribution_samples = num_initial_distribution_samples
        self.initial_distribution = initial_distribution
        self.augment_dataset_with_sim = augment_dataset_with_sim
        self.pre_train_iterations = pre_train_iterations

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
        if (
            self.train_at_observe
            and len(self.memory) > self.batch_size
            and self.algorithm is not None
        ):
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
        if self.pretrain_model:
            self.model_learning_algorithm.learn(
                self.logger, max_iter=self.pre_train_iterations
            )
        if self.learn_model_at_end_episode:
            self.model_learning_algorithm.learn(self.logger)
        if self.train_at_end_episode:
            self.learn()
        if self.simulate:
            self.simulate_policy_on_model()
        super().end_episode()

    def learn(self, memory=None):
        """Learn a policy with the model."""
        #

        def closure():
            """Gradient calculation."""
            if memory is None:
                observation, *_ = self.memory.sample_batch(self.batch_size)
            else:
                observation, *_ = memory.sample_batch(self.batch_size)
            self.optimizer.zero_grad()
            losses = self.algorithm(observation.clone())
            losses.combined_loss.mean().backward()

            torch.nn.utils.clip_grad_norm_(
                self.algorithm.parameters(), self.clip_gradient_val
            )
            return losses

        with DisableGradient(
            self.dynamical_model, self.reward_model, self.termination_model
        ):
            self._learn_steps(closure)

    def _sample_initial_states(self):
        """Get initial states to sample from."""
        # Samples from experience replay empirical distribution.
        if self.num_memory_samples > 0:
            obs, *_ = self.memory.sample_batch(self.num_memory_samples)
            for transform in self.memory.transformations:
                obs = transform.inverse(obs)
            initial_states = obs.state[:, 0, :]  # obs is an n-step return.
            return initial_states
        # Samples from empirical initial state distribution.
        elif self.num_initial_state_samples > 0:
            initial_states = self.initial_states_dataset.sample_batch(
                self.num_initial_state_samples
            )
            # initial_states = torch.cat((initial_states, initial_states_), dim=0)
            return initial_states
        # Samples from initial distribution.
        elif self.num_initial_distribution_samples > 0:
            initial_states = self.initial_distribution.sample(
                (self.num_initial_distribution_samples,)
            )
            return initial_states
            # initial_states = torch.cat((initial_states, initial_states_), dim=0)
        else:
            raise ValueError("At least one has to be larger than zero.")
        # initial_states = initial_states.unsqueeze(0)
        # return initial_states

    def simulate_policy_on_model(self):
        """Evaluate policy on learned model."""
        with torch.no_grad():
            fake_env = FakeEnvironment(
                dynamical_model=self.dynamical_model,
                reward_model=self.reward_model,
                termination_model=self.termination_model,
                initial_state_fn=self._sample_initial_states,
            )
            trajectory = rollout_policy(
                environment=fake_env,
                policy=self.policy,
                num_episodes=1,
                max_steps=self.simulation_max_steps,
                memory=self.memory if self.augment_dataset_with_sim else None,
            )[0]
            sim_returns = torch.cat([obs.reward for obs in trajectory], dim=0).sum(0)
            self.logger.update(
                **{f"Sim-Returns-{i}": returns for i, returns in enumerate(sim_returns)}
            )

    @property
    def pretrain_model(self):
        """Raise flag if learn the model after observe."""
        return (
            self.training
            and self.model_learning_algorithm is not None
            and self.total_steps > self.model_learn_exploration_steps
            and self.total_episodes == self.model_learn_exploration_episodes
            and self.pre_train_iterations > 0
        )

    @property
    def learn_model_at_observe(self):
        """Raise flag if learn the model after observe."""
        return (
            self.training
            and self.model_learning_algorithm is not None
            and self.total_steps > self.model_learn_exploration_steps
            and self.total_episodes > self.model_learn_exploration_episodes
            and self.model_learn_train_frequency > 0
            and self.total_steps % self.model_learn_train_frequency == 0
        )

    @property
    def train_at_observe(self):
        """Return true if model has been learned."""
        return (
            super().train_at_observe
            and self.total_episodes > self.model_learn_exploration_episodes + 1
        )

    @property
    def learn_model_at_end_episode(self):
        """Raise flag to learn the model at end of an episode."""
        return (
            self.training
            and self.model_learning_algorithm is not None
            and self.total_steps > self.model_learn_exploration_steps
            and self.total_episodes > self.model_learn_exploration_episodes
            and self.model_learn_num_rollouts > 0
            and (self.total_episodes + 1) % self.model_learn_num_rollouts == 0
        )

    @property
    def train_at_end_episode(self):
        """Return true if model has been learned."""
        return (
            super().train_at_end_episode
            and self.total_episodes > self.model_learn_exploration_episodes + 1
        )

    @property
    def simulate(self):
        """Flag that indicates whether to simulate."""
        return (
            self.simulation_frequency
            and (self.train_episodes % self.simulation_frequency) == 0
        )

    @classmethod
    def default(
        cls,
        environment,
        dynamical_model=None,
        reward_model=None,
        termination_model=None,
        num_epochs=50,
        model_lr=5e-4,
        l2_reg=1e-4,
        calibrate=True,
        *args,
        **kwargs,
    ):
        """Get a default model-based agent."""
        if dynamical_model is None:
            dynamical_model = TransformedModel.default(environment, *args, **kwargs)
        if reward_model is None:
            try:
                reward_model = environment.env.reward_model()
            except AttributeError:
                reward_model = TransformedModel.default(
                    environment,
                    model_kind="rewards",
                    transformations=dynamical_model.transformations,
                )
        if termination_model is None:
            try:
                termination_model = environment.env.termination_model()
            except AttributeError:
                pass
        params = list(
            chain(
                [p for p in dynamical_model.parameters() if p.requires_grad],
                [p for p in reward_model.parameters() if p.requires_grad],
            )
        )
        if len(params):
            model_optimizer = Adam(params, lr=model_lr, weight_decay=l2_reg)

            model_learning_algorithm = ModelLearningAlgorithm(
                dynamical_model=dynamical_model,
                reward_model=reward_model,
                termination_model=termination_model,
                num_epochs=num_epochs,
                model_optimizer=model_optimizer,
                calibrate=calibrate,
            )
        else:
            model_learning_algorithm = None

        return super().default(
            environment,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            termination_model=termination_model,
            model_learning_algorithm=model_learning_algorithm,
            *args,
            **kwargs,
        )
