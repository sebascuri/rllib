"""Python Script Template."""
import copy

from rllib.agent import MPCAgent
from rllib.algorithms.mpc import CEMShooting, MPCSolver, MPPIShooting, RandomShooting
from rllib.dataset.experience_replay import ExperienceReplay
from rllib.environment import GymEnvironment
from rllib.model.environment_model import EnvironmentModel
from rllib.util.training.agent_training import evaluate_agent

SEED = 0
MAX_STEPS = 200
ENVIRONMENT = ["VPendulum-v0", "VContinuous-CartPole-v0"][1]

env = GymEnvironment(ENVIRONMENT, SEED)
env_model = copy.deepcopy(env)
env_model.reset()
dynamical_model = EnvironmentModel(env_model, model_kind="dynamics")
reward_model = EnvironmentModel(env_model, model_kind="rewards")
termination_model = EnvironmentModel(env_model, model_kind="termination")
GAMMA = 0.99
horizon = 50
num_iter = 5
num_samples = 400
num_elites = 5
num_steps = horizon
solver = "cem_shooting"
kappa = 1.0
betas = [0.2, 0.8, 0]
warm_start = True
num_cpu = 1

memory = ExperienceReplay(max_len=2000, num_steps=1)
value_function = None

if solver == "random_shooting":
    mpc_solver = RandomShooting(
        dynamical_model=dynamical_model,
        reward_model=reward_model,
        horizon=horizon,
        gamma=GAMMA,
        num_samples=num_samples,
        num_elites=num_elites,
        termination_model=termination_model,
        terminal_reward=value_function,
        warm_start=warm_start,
        default_action="mean",
        num_cpu=num_cpu,
    )  # type: MPCSolver
elif solver == "cem_shooting":
    mpc_solver = CEMShooting(
        dynamical_model=dynamical_model,
        reward_model=reward_model,
        horizon=horizon,
        gamma=GAMMA,
        num_iter=num_iter,
        num_samples=num_samples,
        num_elites=num_elites,
        termination_model=termination_model,
        terminal_reward=value_function,
        warm_start=warm_start,
        default_action="mean",
        num_cpu=num_cpu,
    )
elif solver == "mppi_shooting":
    mpc_solver = MPPIShooting(
        dynamical_model=dynamical_model,
        reward_model=reward_model,
        horizon=horizon,
        gamma=GAMMA,
        num_iter=num_iter,
        kappa=kappa,
        filter_coefficients=betas,
        num_samples=num_samples,
        termination_model=termination_model,
        terminal_reward=value_function,
        warm_start=warm_start,
        default_action="mean",
        num_cpu=num_cpu,
    )
else:
    raise NotImplementedError

agent = MPCAgent(mpc_solver=mpc_solver)
evaluate_agent(agent, environment=env, num_episodes=1, max_steps=MAX_STEPS, render=True)
