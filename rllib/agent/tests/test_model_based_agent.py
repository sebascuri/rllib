import copy

import numpy as np
import pytest
import torch

from rllib.agent import BPTTAgent, DynaAgent, MPCAgent, MVEAgent, STEVEAgent, SVGAgent
from rllib.algorithms.mpc import CEMShooting, MPPIShooting, RandomShooting
from rllib.environment import GymEnvironment
from rllib.model.environment_model import EnvironmentModel
from rllib.util.training.agent_training import evaluate_agent, train_agent

MAX_STEPS = 25
NUM_EPISODES = 2
SEED = 0


def rollout_agent(environment, agent):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_agent(
        agent,
        environment,
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEPS,
        plot_flag=False,
    )
    evaluate_agent(agent, environment, 1, MAX_STEPS, render=False)
    agent.logger.delete_directory()  # Cleanup directory.


class TestBPTTAgents(object):
    @pytest.fixture(params=["LunarLanderContinuous-v2"], scope="class")
    def environment(self, request):
        return GymEnvironment(request.param, SEED)

    @pytest.fixture(params=[BPTTAgent, SVGAgent], scope="class")
    def agent(self, request):
        return request.param

    def test_continuous_agent(self, environment, agent):
        agent = agent.default(
            environment,
            num_iter=2,
            num_epochs=2,
            exploration_steps=0,
            exploration_episodes=0,
            simulation_max_steps=10,
        )
        rollout_agent(environment, agent)


class TestDerivedMBAgents(object):
    @pytest.fixture(params=["LunarLanderContinuous-v2"], scope="class")
    def environment(self, request):
        return GymEnvironment(request.param, SEED)

    @pytest.fixture(params=[1, 4], scope="class")
    def num_model_steps(self, request):
        return request.param

    @pytest.fixture(params=["DPG", "TD3", "SAC", "MPO", "VMPO"], scope="class")
    def base_agent(self, request):
        return request.param

    @pytest.fixture(params=[DynaAgent, STEVEAgent, MVEAgent], scope="class")
    def extender(self, request):
        return request.param

    def test_continuous_agent(self, environment, base_agent, extender, num_model_steps):
        agent = extender.default(
            environment,
            base_agent_name=base_agent,
            num_model_steps=num_model_steps,
            num_particles=2,
            num_iter=2,
            num_epochs=2,
            td_k=True,
            exploration_steps=0,
            exploration_episodes=0,
            simulation_max_steps=10,
        )
        rollout_agent(environment, agent)

    def test_mve_not_td_k(self, environment, base_agent, num_model_steps):
        agent = MVEAgent.default(
            environment,
            base_agent_name=base_agent,
            num_model_steps=num_model_steps,
            num_particles=2,
            num_iter=2,
            num_epochs=2,
            td_k=False,
            exploration_steps=0,
            exploration_episodes=0,
        )
        rollout_agent(environment, agent)


class TestMPCAgent(object):
    ENVIRONMENT = "VContinuous-CartPole-v0"
    GAMMA = 0.99
    HORIZON = 5
    NUM_ITER = 5
    NUM_PARTICLES = 50
    NUM_ELITES = 5
    KAPPA = 1.0
    BETAS = (0.2, 0.8, 0.0)
    MAX_ITER = 5

    def init(self):
        self.env = GymEnvironment(self.ENVIRONMENT, SEED)
        self.env_model = copy.deepcopy(self.env)
        self.env_model.reset()
        self.dynamical_model = EnvironmentModel(self.env_model, model_kind="dynamics")
        self.reward_model = EnvironmentModel(self.env_model, model_kind="rewards")
        self.termination_model = EnvironmentModel(
            self.env_model, model_kind="termination"
        )
        super().__init__()

    @pytest.fixture(
        params=["random_shooting", "cem_shooting", "mppi_shooting"], scope="class"
    )
    def solver(self, request):
        return request.param

    @pytest.fixture(params=[True, False], scope="class")
    def warm_start(self, request):
        return request.param

    @pytest.fixture(params=["mean", "zero", "constant"], scope="class")
    def default_action(self, request):
        return request.param

    def get_solver(self, solver_, warm_start_, num_cpu_, default_action_):
        if solver_ == "random_shooting":
            mpc_solver = RandomShooting(
                dynamical_model=self.dynamical_model,
                reward_model=self.reward_model,
                horizon=self.HORIZON,
                gamma=1.0,
                num_particles=self.NUM_PARTICLES,
                num_elites=self.NUM_ELITES,
                termination_model=self.termination_model,
                terminal_reward=None,
                warm_start=warm_start_,
                default_action=default_action_,
                num_cpu=num_cpu_,
            )
        elif solver_ == "cem_shooting":
            mpc_solver = CEMShooting(
                dynamical_model=self.dynamical_model,
                reward_model=self.reward_model,
                horizon=self.HORIZON,
                gamma=1.0,
                num_iter=self.NUM_ITER,
                num_particles=self.NUM_PARTICLES,
                num_elites=self.NUM_ELITES,
                termination_model=self.termination_model,
                terminal_reward=None,
                warm_start=warm_start_,
                default_action=default_action_,
                num_cpu=num_cpu_,
            )
        elif solver_ == "mppi_shooting":
            mpc_solver = MPPIShooting(
                dynamical_model=self.dynamical_model,
                reward_model=self.reward_model,
                horizon=self.HORIZON,
                gamma=1.0,
                num_iter=self.NUM_ITER,
                kappa=self.KAPPA,
                filter_coefficients=self.BETAS,
                num_particles=self.NUM_PARTICLES,
                termination_model=self.termination_model,
                terminal_reward=None,
                warm_start=warm_start_,
                default_action=default_action_,
                num_cpu=num_cpu_,
            )
        else:
            raise NotImplementedError
        return mpc_solver

    def run_agent(self, mpc_solver):
        agent = MPCAgent(
            mpc_solver=mpc_solver, exploration_steps=0, exploration_episodes=0
        )
        evaluate_agent(
            agent,
            environment=self.env,
            num_episodes=1,
            max_steps=self.MAX_ITER,
            render=False,
        )
        agent.logger.delete_directory()  # Cleanup directory.

    def test_mpc_solvers(self, solver):
        self.init()
        mpc_solver = self.get_solver(solver, True, 1, "mean")
        self.run_agent(mpc_solver)

    def test_mpc_warm_start(self, solver, warm_start):
        self.init()
        mpc_solver = self.get_solver(solver, warm_start, 1, "mean")
        self.run_agent(mpc_solver)

    def test_mpc_default_action(self, solver, default_action):
        self.init()
        mpc_solver = self.get_solver(solver, True, 1, default_action)
        self.run_agent(mpc_solver)
