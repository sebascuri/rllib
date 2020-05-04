"""Model Based Agent."""

import gpytorch
from gym.utils import colorize
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .abstract_agent import AbstractAgent
from rllib.dataset.datatypes import Observation
from rllib.dataset.experience_replay import BootstrapExperienceReplay, ExperienceReplay
from rllib.dataset.utilities import stack_list_of_tuples

from rllib.model import ExactGPModel
from rllib.util.gaussian_processes import SparseGP
from rllib.util.neural_networks.utilities import disable_gradient
from rllib.util.rollout import rollout_model
from rllib.util.utilities import tensor_to_distribution
from rllib.util.training import train_model
from rllib.util.value_estimation import mb_return


class ModelBasedAgent(AbstractAgent):
    """Implementation of a Model Based RL Agent.

    Parameters
    ----------
    env_name: str.
        Environment name.
    dynamical_model: AbstractModel.
        Fixed or learnable dynamical model.
    reward_model: AbstractReward.
        Fixed or learnable reward model.
    model_optimizer: Optim
        Optimizer for dynamical_model and reward_model.
    policy: AbstractPolicy.
        Fixed or learnable policy.
    value_function: AbstractValueFunction, optional. (default: None).
        Fixed or learnable value function used for planning.
    termination: Callable, optional. (default: None).
        Fixed or learnable termination condition.

    plan_horizon: int, optional. (default: 0).
        If plan_horizon = 0: the agent returns a sample from the current policy when
        'agent.act(state)' is called.
        If plan_horizon > 0: the agent uses the model to plan for plan_horizon steps and
        returns the action that optimizes the plan.
    plan_samples: int, optional. (default: 1).
        Number of samples used to solve the planning problem.
    plan_elite: int, optional. (default: 1).
        Number of elite samples used to return the best action.

    model_learn_num_iter: int, optional. (default: 0).
        Number of iteration for model learning.
    model_learn_batch_size: int, optional. (default: 64).
        Batch size of model learning algorithm.
    max_memory: int, optional. (default: 10000).
        Maximum size of data set.

    policy_opt_num_iter: int, optional. (default: 0).
        Number of iterations for policy optimization.
    policy_opt_batch_size: int, optional. (default: model_learn_batch_size).
        Batch size of policy optimization algorithm.

    sim_num_steps: int, optional. (default: 20).
        Number of simulation steps.
    sim_initial_states_num_trajectories: int, optional. (default: 8).
        Number of simulation trajectories that start from a sample of the empirical
        distribution.
    sim_initial_dist_num_trajectories: int, optional. (default: 0).
        Number of simulation trajectories that start from a sample of a selected initial
        distribution.
    sim_memory_num_trajectories: int, optional. (default: 0).
        Number of simulation trajectories that start from a sample of the dataset.
    sim_refresh_interval: int, optional.
        Number of policy optimization steps.
    sim_num_subsample: int, optional. (default: 1).
        Add one out of `sim_num_subsample' samples to the data set.

    initial_distribution: Distribution, optional. (default: None).
        Initial state distribution.
    gamma: float, optional. (default: 0.99).
    exploration_steps: int, optional. (default: 0).
    exploration_episodes: int, optional. (default: 0).
    comment: str, optional. (default: '').
    """

    def __init__(self, env_name, dynamical_model, reward_model, model_optimizer,
                 policy,
                 value_function=None,
                 termination=None,
                 plan_horizon=0,
                 plan_samples=1,
                 plan_elite=1,
                 model_learn_num_iter=0,
                 model_learn_batch_size=64,
                 max_memory=10000,
                 policy_opt_num_iter=0,
                 policy_opt_batch_size=None,
                 sim_num_steps=20,
                 sim_initial_states_num_trajectories=8,
                 sim_initial_dist_num_trajectories=0,
                 sim_memory_num_trajectories=0,
                 sim_refresh_interval=1,
                 sim_num_subsample=1,
                 initial_distribution=None,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0, comment=''):
        super().__init__(env_name, gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes, comment=comment)
        self.dynamical_model = dynamical_model
        self.reward_model = reward_model
        self.termination = termination
        self.model_optimizer = model_optimizer
        self.value_function = value_function

        self.model_learn_num_iter = model_learn_num_iter
        self.model_learn_batch_size = model_learn_batch_size

        self.policy = policy
        self.plan_horizon = plan_horizon
        self.plan_samples = plan_samples
        self.plan_elite = plan_elite

        if hasattr(dynamical_model.base_model, 'num_heads'):
            num_heads = dynamical_model.base_model.num_heads
        else:
            num_heads = 1

        self.dataset = BootstrapExperienceReplay(
            max_len=max_memory, transformations=dynamical_model.forward_transformations,
            num_bootstraps=num_heads)
        self.sim_dataset = ExperienceReplay(
            max_len=max_memory, transformations=dynamical_model.forward_transformations)

        self.policy_opt_num_iter = policy_opt_num_iter
        if policy_opt_batch_size is None:  # set the same batch size as in model learn.
            policy_opt_batch_size = self.model_learn_batch_size
        self.policy_opt_batch_size = policy_opt_batch_size

        self.sim_trajectory = None

        self.sim_num_steps = sim_num_steps
        self.sim_initial_states_num_trajectories = sim_initial_states_num_trajectories
        self.sim_initial_dist_num_trajectories = sim_initial_dist_num_trajectories
        self.sim_memory_num_trajectories = sim_memory_num_trajectories
        self.sim_refresh_interval = sim_refresh_interval
        self.sim_num_subsample = sim_num_subsample
        self.initial_distribution = initial_distribution
        self.initial_states = torch.tensor(float('nan'))
        self.new_episode = True

    def act(self, state):
        """Ask the agent for an action to interact with the environment.

        If the plan horizon is zero, then it just samples an action from the policy.
        If the plan horizon > 0, then is plans with the current model.
        """
        if self.plan_horizon == 0:
            action = super().act(state)
        else:
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.get_default_dtype())
            policy = tensor_to_distribution(self.policy(state))
            self.pi = policy
            action = self._plan(state).detach().numpy()

        return action[..., :self.dynamical_model.dim_action]

    def observe(self, observation):
        """Observe a new transition.

        If the episode is new, add the initial state to the state transitions.
        Add the transition to the data set.
        """
        super().observe(observation)
        self.dataset.append(observation)
        if self.new_episode:
            initial_state = observation.state.unsqueeze(0)
            if torch.isnan(self.initial_states).any():
                self.initial_states = initial_state
            else:
                self.initial_states = torch.cat((self.initial_states, initial_state))
            self.new_episode = False

    def start_episode(self):
        """See `AbstractAgent.start_episode'."""
        super().start_episode()
        self.new_episode = True

    def end_episode(self):
        """See `AbstractAgent.end_episode'.

        If the agent is training, and the base model is a GP Model, then add the
        transitions to the GP, and summarize and sparsify the GP Model.

        Then train the agent.
        """
        if self._training:
            if isinstance(self.dynamical_model.base_model, ExactGPModel):
                observation = stack_list_of_tuples(self.last_trajectory)
                for transform in self.dataset.transformations:
                    observation = transform(observation)
                print(colorize('Add data to GP Model', 'yellow'))
                self.dynamical_model.base_model.add_data(
                    observation.state, observation.action, observation.next_state)

                print(colorize('Summarize GP Model', 'yellow'))
                self.dynamical_model.base_model.summarize_gp()

                for i, gp in enumerate(self.dynamical_model.base_model.gp):
                    self.logger.update(**{f'gp{i} num inputs': len(gp.train_targets)})

                    if isinstance(gp, SparseGP):
                        self.logger.update(**{
                            f'gp{i} num inducing inputs': gp.xu.shape[0]})

            self._train()
        super().end_episode()

    def _plan(self, state):
        """Plan with current model and policy by (approximately) solving MPC.

        To solve MPC, the policy is sampled to guide random shooting.
        The average of the top `self.plan_elite' samples is returned.
        """
        self.dynamical_model.eval()
        value, trajectory = mb_return(
            state, dynamical_model=self.dynamical_model,
            reward_model=self.reward_model, policy=self.policy,
            num_steps=self.plan_horizon, gamma=self.gamma,
            num_samples=self.plan_samples, value_function=self.value_function,
            termination=self.termination)
        actions = stack_list_of_tuples(trajectory).action
        idx = torch.topk(value, k=self.plan_elite, largest=True)[1]
        return actions[0, idx].mean(0)  # Return first action.

    def _train(self) -> None:
        """Train the agent.

        This consists of two steps:
            Step 1: Train Model with new data.
                Calls self._train_model().
            Step 2: Optimize policy with simulated data.
                Calls self._simulate_and_optimize_policy().
        """
        # Step 1: Train Model with new data.
        self._train_model()
        if self.total_steps < self.exploration_steps or (
                self.total_episodes < self.exploration_episodes):
            return

        # Step 2: Optimize policy with simulated data.
        self._simulate_and_optimize_policy()

    def _train_model(self):
        """Train the models.

        This consists of different steps:
            Step 1: Train dynamical model.
            Step 2: TODO Train the reward model.
            Step 3: TODO Train the initial distribution model.
        """
        print(colorize('Training Model', 'yellow'))
        loader = DataLoader(self.dataset, batch_size=self.model_learn_batch_size,
                            shuffle=True)
        train_model(self.dynamical_model.base_model, train_loader=loader,
                    max_iter=self.model_learn_num_iter, optimizer=self.model_optimizer,
                    logger=self.logger)

    def _simulate_and_optimize_policy(self):
        """Simulate the model and optimize the policy with the learned data.

        This consists of two steps:
            Step 1: Simulate trajectories with the model.
                Calls self._simulate_model().
            Step 2: Implement a model free RL method that optimizes the policy.
                Calls self._optimize_policy(). To be implemented by a Base Class.
        """
        print(colorize('Optimizing Policy with Model Data', 'yellow'))
        self.dynamical_model.eval()
        with disable_gradient(self.dynamical_model), gpytorch.settings.fast_pred_var():
            for i in tqdm(range(self.policy_opt_num_iter)):
                # Step 1: Compute the state distribution
                if (i + 1) % self.sim_refresh_interval == 0:
                    with torch.no_grad():
                        self._simulate_model()

                    average_return = self.sim_trajectory.reward.sum(0).mean().item()
                    average_scale = torch.diagonal(
                            self.sim_trajectory.next_state_scale_tril, dim1=-1, dim2=-2
                        ).sum(-1).sum(0).mean().item()
                    self.logger.update(model_return=average_return)
                    self.logger.update(total_scale=average_scale)
                    # self._trajectory_to_experience_replay()

                # Step 2: Optimize policy
                self._optimize_policy()

    def _trajectory_to_experience_replay(self):
        """Convert trajectories to experience replay."""
        total_num_trajectories = self.sim_trajectory.state.shape[2]
        trajectories = [Observation(*[(a[:, 0, i_traj] if a.dim() > 2 else a)
                                      for a in self.sim_trajectory])
                        for i_traj in range(total_num_trajectories)]

        for trajectory in trajectories:
            for i in range(trajectory.state.shape[0]):
                if i % self.sim_num_subsample == 0:
                    observation = Observation(*[a[i] for a in trajectory])
                    self.sim_dataset.append(observation)

    def _simulate_model(self):
        """Simulate the model.

        The simulation is initialized by concatenating samples from:
            - The empirical initial state distribution.
            - A learned or fixed initial state distribution.
            - The empirical state distribution.
        """
        # Samples from empirical initial state distribution.
        idx = torch.randint(self.initial_states.shape[0],
                            (self.sim_initial_states_num_trajectories,))
        initial_states = self.initial_states[idx]

        # Samples from initial distribution.
        if self.sim_initial_dist_num_trajectories > 0:
            initial_states_ = self.initial_distribution.sample((
                self.sim_initial_dist_num_trajectories,))
            initial_states = torch.cat((initial_states, initial_states_), dim=0)

        # Samples from experience replay empirical distribution.
        if self.sim_memory_num_trajectories > 0:
            obs, *_ = self.dataset.get_batch(self.sim_memory_num_trajectories)
            for transform in self.dataset.transformations:
                obs = transform.inverse(obs)
            initial_states_ = obs.state[:, 0, :]  # obs is an n-step return.
            initial_states = torch.cat((initial_states, initial_states_), dim=0)

        initial_states = initial_states.unsqueeze(0)

        trajectory = rollout_model(dynamical_model=self.dynamical_model,
                                   reward_model=self.reward_model,
                                   policy=self.policy,
                                   initial_state=initial_states,
                                   max_steps=self.sim_num_steps,
                                   termination=self.termination)

        self.sim_trajectory = stack_list_of_tuples(trajectory)

    def _optimize_policy(self):
        """Optimize the policy.

        The policy here is optimized using simulated data.
        Child classes should implement this method if needed.
        """
        pass
