"""Template for a Model Based Agent.

A model based agent has three behaviors:
- It learns models from data collected from the environment.
- It optimizes policies with simulated data from the models.
- It plans with the model and policies (as guiding sampler).
"""

import torch
from gym.utils import colorize
from tqdm import tqdm

from rllib.agent.abstract_agent import AbstractAgent
from rllib.algorithms.mpc.policy_shooting import PolicyShooting
from rllib.dataset.datatypes import Observation
from rllib.dataset.experience_replay import ExperienceReplay, StateExperienceReplay
from rllib.dataset.utilities import stack_list_of_tuples
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
        policy_learning_algorithm=None,
        model_learning_algorithm=None,
        planning_algorithm=None,
        simulation_algorithm=None,
        num_simulation_iterations=0,
        num_rollouts=1,
        learn_from_real=False,
        thompson_sampling=False,
        memory=None,
        training_verbose=True,
        *args,
        **kwargs,
    ):
        super().__init__(
            num_rollouts=num_rollouts,
            training_verbose=training_verbose,
            *args,
            **kwargs,
        )
        self.algorithm = policy_learning_algorithm
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
            memory = ExperienceReplay(max_len=50000, num_steps=0)
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
            policy = tensor_to_distribution(self.policy(state), **self.dist_params)
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
        self.memory.append(observation)
        super().observe(observation)

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
        if self._training:  # training mode.
            self.learn()
        super().end_episode()

    def learn(self):
        """Train the agent.

        This consists of two steps:
            Step 1: Train Model with new data.
                Calls self.learn_model().
            Step 2: Optimize policy with simulated data.
                Calls self.simulate_and_learn_policy().
        """
        # Step 1: Train Model with new data.
        self.model_learning_algorithm.learn(self.last_trajectory, self.logger)

        if (
            self.total_steps >= self.exploration_steps  # enough steps.
            and self.total_episodes >= self.exploration_episodes  # enough episodes.
            and self.num_rollouts > 0  # train once the episode ends.
            and (self.total_episodes + 1) % self.num_rollouts == 0  # correct steps.
        ):
            # Step 2: Optimize policy with simulated data.
            if self.learn_from_sim:
                print(colorize("Optimizing Policy with Simulated Data", "yellow"))
                self.simulate_and_learn_policy()

            # Step 3: Optimize policy with real data.
            if self.learn_from_real:
                print(colorize("Optimizing Policy with Real Data", "yellow"))
                self.learn_policy_from_real_data()

    def simulate_and_learn_policy(self):
        """Simulate the model and optimize the policy with the learned data.

        This consists of two steps:
            Step 1: Simulate trajectories with the model.
                Calls self.simulate_model().
            Step 2: Implement a model free RL method that optimizes the policy.
                Calls self.learn_policy(). To be implemented by a Base Class.
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

                    trajectory = self.simulation_algorithm.simulate(
                        initial_states, self.policy
                    )
                    self.log_trajectory(trajectory)

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

    def log_trajectory(self, trajectory):
        """Log simulated trajectory."""
        if self.logger is None:
            return
        trajectory = stack_list_of_tuples(trajectory)
        scale = torch.diagonal(trajectory.next_state_scale_tril, dim1=-1, dim2=-2)
        self.logger.update(
            sim_entropy=trajectory.entropy.mean().item(),
            sim_return=trajectory.reward.sum(0).mean().item(),
            sim_scale=scale.square().sum(-1).sum(0).mean().sqrt().item(),
            sim_max_state=trajectory.state.abs().max().item(),
            sim_max_action=trajectory.action.abs().max().item(),
        )
        for key, value in self.simulation_algorithm.reward_model.info.items():
            self.logger.update(**{f"sim_{key}": value})
