"""Python Script Template."""
from rllib.agent import MVEAgent as agent_
from rllib.environment import GymEnvironment
from rllib.util.training.agent_training import train_agent
from rllib.util.utilities import set_random_seed
from exps.half_cheetah.utils import HalfCheetahReward

MAX_STEPS = 1000
NUM_EPISODES = 100
ACTION_COST = 0.1
SEED = 0

set_random_seed(SEED)
env = GymEnvironment("HalfCheetah-fullobs-v0", seed=SEED, action_cost=ACTION_COST)

agent = agent_.default(
    environment=env,
    # reward_model=HalfCheetahReward(action_cost=ACTION_COST),
    exploration_episodes=10,
    model_learn_exploration_episodes=5,
    num_steps=3,
)
agent.algorithm.td_k = False
train_agent(
    agent, env, max_steps=MAX_STEPS, num_episodes=NUM_EPISODES, print_frequency=1,
)
