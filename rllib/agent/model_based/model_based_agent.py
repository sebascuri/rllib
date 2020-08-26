"""Template for a Model Based Agent.

A model based agent has three behaviors:
- It learns models from data collected from the environment.
- It optimizes policies with simulated data from the models.
- It plans with the model and policies (as guiding sampler).
"""

import gpytorch
import torch
from gym.utils import colorize
from tqdm import tqdm

from rllib.agent.abstract_agent import AbstractAgent
from rllib.algorithms.mpc.policy_shooting import PolicyShooting
from rllib.dataset.datatypes import Observation
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.policy.derived_policy import DerivedPolicy
from rllib.policy.mpc_policy import MPCPolicy
from rllib.util.neural_networks.utilities import DisableGradient
from rllib.util.utilities import tensor_to_distribution


class ModelBasedAgent(AbstractAgent):
    """Implementation of a Model Based RL Agent."""

    def __init__(
        self,
        policy_learning_algorithm=None,
        model_learning_algorithm=None,
        planning_algorithm=None,
        simulation_algorithm=None,
        num_simulation_iterations=0,
        thompson_sampling=False,
        learn_from_real=False,
        *args,
        **kwargs,
    ):
        super().__init__(train_frequency=0, num_rollouts=0, *args, **kwargs)
        self.policy_learning_algorithm = policy_learning_algorithm
        self.planning_algorithm = planning_algorithm
        self.model_learning_algorithm = model_learning_algorithm
        self.simulation_algorithm = simulation_algorithm
        dynamical_model, reward_model, policy = None, None, None

        if policy_learning_algorithm is not None:
            policy = policy_learning_algorithm.policy
        elif planning_algorithm is not None:
            policy = MPCPolicy(self.planning_algorithm)
        else:
            raise NotImplementedError

        for alg in [
            policy_learning_algorithm,
            planning_algorithm,
            model_learning_algorithm,
            simulation_algorithm,
        ]:
            if alg is not None:
                dynamical_model, reward_model = alg.dynamical_model, alg.reward_model
                break

        self.dynamical_model = dynamical_model
        self.reward_model = reward_model

        self.policy = DerivedPolicy(policy, self.dynamical_model.base_model.dim_action)
        self.num_simulation_iterations = num_simulation_iterations
        self.learn_from_real = learn_from_real
        self.thompson_sampling = thompson_sampling

        if self.thompson_sampling:
            self.dynamical_model.set_prediction_strategy("posterior")

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
        if self._training:
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

        if self.total_steps < self.exploration_steps or (
            self.total_episodes < self.exploration_episodes
        ):
            return

        # Step 2: Optimize policy with simulated data.
        self.simulate_and_learn_policy()

        # Step 3: Optimize policy with real data.
        if self.learn_from_real:
            self.learn_policy_from_real_data()

    def simulate_and_learn_policy(self):
        """Simulate the model and optimize the policy with the learned data.

        This consists of two steps:
            Step 1: Simulate trajectories with the model.
                Calls self.simulate_model().
            Step 2: Implement a model free RL method that optimizes the policy.
                Calls self.learn_policy(). To be implemented by a Base Class.
        """
        print(colorize("Optimizing Policy with Model Data", "yellow"))
        self.dynamical_model.eval()
        # self.simulation_algorithm.dataset.reset()

        with DisableGradient(
            self.dynamical_model, self.reward_model
        ), gpytorch.settings.fast_pred_var():
            for _ in tqdm(range(self.num_simulation_iterations)):
                # Step 1: Simulate the state distribution
                with torch.no_grad():
                    self.policy.reset()  # TODO: Add goal distribution.
                    initial_states = self.simulation_algorithm.get_initial_states(
                        self.model_learning_algorithm.initial_states_dataset,
                        self.model_learning_algorithm.dataset,
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
            observation = Observation(
                state=self.simulation_algorithm.dataset.get_batch(self.batch_size)
            )
            self.optimizer.zero_grad()
            losses_ = self.policy_learning_algorithm(observation)
            losses_.combined_loss.mean().backward()

            torch.nn.utils.clip_grad_norm_(
                self.policy_learning_algorithm.parameters(), self.clip_gradient_val
            )

            return losses_

        self._learn_steps(closure)

    def learn_policy_from_real_data(self):
        """Learn policy using real transitions and use the model to predict targets."""
        #

        def closure():
            """Gradient calculation."""
            observation = self.model_learning_algorithm.dataset.get_batch(
                self.batch_size
            )
            self.optimizer.zero_grad()
            losses_ = self.policy_learning_algorithm(observation)
            losses_.combined_loss.mean().backward()

            torch.nn.utils.clip_grad_norm_(
                self.policy_learning_algorithm.parameters(), self.clip_gradient_val
            )

            return losses_

        self._learn_steps(closure)

    def log_trajectory(self, trajectory):
        """Log simulated trajectory."""
        if self.logger is None:
            return
        trajectory = stack_list_of_tuples(trajectory)
        average_return = trajectory.reward.sum(0).mean().item()
        average_scale = (
            torch.diagonal(trajectory.next_state_scale_tril, dim1=-1, dim2=-2)
            .square()
            .sum(-1)
            .sum(0)
            .mean()
            .sqrt()
            .item()
        )
        self.logger.update(sim_entropy=trajectory.entropy.mean().item())
        self.logger.update(sim_return=average_return)
        self.logger.update(sim_scale=average_scale)
        self.logger.update(sim_max_state=trajectory.state.abs().max().item())
        self.logger.update(sim_max_action=trajectory.action.abs().max().item())
        try:
            r_ctrl = self.simulation_algorithm.reward_model.reward_ctrl.mean()
            r_state = self.simulation_algorithm.reward_model.reward_state.mean()
            self.logger.update(sim_reward_ctrl=r_ctrl.detach().item())
            self.logger.update(sim_reward_state=r_state.detach().item())
        except AttributeError:
            pass
        try:
            r_o = self.simulation_algorithm.reward_model.reward_dist_to_obj
            r_g = self.simulation_algorithm.reward_model.reward_dist_to_goal
            self.logger.update(sim_reward_dist_to_obj=r_o.mean().detach().item())
            self.logger.update(sim_reward_dist_to_goal=r_g.mean().detach().item())
        except AttributeError:
            pass
