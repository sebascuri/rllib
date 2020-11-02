"""Working example of Q-LEARNING."""
import numpy as np
import torch.optim

from rllib.agent import DDQNAgent, DQNAgent  # noqa: F401
from rllib.dataset import EXP3ExperienceReplay  # noqa: F401
from rllib.dataset import PrioritizedExperienceReplay  # noqa: F401
from rllib.dataset import ExperienceReplay
from rllib.environment import GymEnvironment
from rllib.policy import EpsGreedy
from rllib.util.neural_networks.utilities import init_head_bias  # noqa: F401
from rllib.util.neural_networks.utilities import init_head_weight, zero_bias
from rllib.util.parameter_decay import ExponentialDecay, LinearGrowth  # noqa: F401
from rllib.util.training.agent_training import evaluate_agent, train_agent
from rllib.value_function import NNQFunction

ENVIRONMENT = ["NChain-v0", "CartPole-v0"][1]

NUM_EPISODES = 100
MAX_STEPS = 200
TARGET_UPDATE_FREQUENCY = 4
TARGET_UPDATE_TAU = 0
MEMORY_MAX_SIZE = 5000
BATCH_SIZE = 64
LEARNING_RATE = 1e-2
MOMENTUM = 0.1
WEIGHT_DECAY = 1e-4
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500
LAYERS = [64, 64]
SEED = 1
MEMORY = "ER"

torch.manual_seed(SEED)
np.random.seed(SEED)

environment = GymEnvironment(ENVIRONMENT, SEED)
q_function = NNQFunction(
    dim_state=environment.dim_state,
    dim_action=environment.dim_action,
    num_states=environment.num_states,
    num_actions=environment.num_actions,
    layers=LAYERS,
    non_linearity="ReLU",
    tau=TARGET_UPDATE_TAU,
)

zero_bias(q_function)
init_head_weight(q_function)
# init_head_bias(q_function, offset=(1 - GAMMA ** 200) / (1 - GAMMA))
policy = EpsGreedy(q_function, ExponentialDecay(EPS_START, EPS_END, EPS_DECAY))


optimizer = torch.optim.Adam(
    q_function.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
criterion = torch.nn.MSELoss

if MEMORY == "PER":
    memory = PrioritizedExperienceReplay(
        max_len=MEMORY_MAX_SIZE, beta=LinearGrowth(0.8, 1.0, 0.001)
    )
elif MEMORY == "EXP3":
    memory = EXP3ExperienceReplay(max_len=MEMORY_MAX_SIZE, alpha=0.001, beta=0.1)
elif MEMORY == "ER":
    memory = ExperienceReplay(max_len=MEMORY_MAX_SIZE, num_steps=0)
else:
    raise NotImplementedError(f"{MEMORY} not implemented.")

agent = DDQNAgent(
    critic=q_function,
    policy=policy,
    criterion=criterion,
    optimizer=optimizer,
    memory=memory,
    num_iter=1,
    train_frequency=1,
    batch_size=BATCH_SIZE,
    target_update_frequency=TARGET_UPDATE_FREQUENCY,
    gamma=GAMMA,
    clip_gradient_val=1.0,
)

train_agent(agent, environment, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS)
evaluate_agent(agent, environment, num_episodes=1, max_steps=MAX_STEPS)
