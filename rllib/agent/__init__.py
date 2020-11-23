from .abstract_agent import AbstractAgent
from .bandit import GPUCBAgent
from .model_based import (
    BPTTAgent,
    DataAugmentationAgent,
    DynaAgent,
    ModelBasedAgent,
    MPCAgent,
    MVEAgent,
    STEVEAgent,
    SVGAgent,
)
from .off_policy import (
    DDQNAgent,
    DPGAgent,
    DQNAgent,
    ISERAgent,
    MPOAgent,
    QLearningAgent,
    REPSAgent,
    SACAgent,
    SoftQLearningAgent,
    SVG0Agent,
    TD3Agent,
    VMPOAgent,
)
from .on_policy import (
    A2CAgent,
    ActorCriticAgent,
    ExpectedActorCriticAgent,
    ExpectedSARSAAgent,
    GAACAgent,
    PPOAgent,
    REINFORCEAgent,
    SARSAAgent,
    TRPOAgent,
)
from .random_agent import RandomAgent

MODEL_FREE = [
    "A2C",
    "ActorCritic",
    "DDQN",
    "DPG",
    "DQN",
    "ExpectedActorCritic",
    "ExpectedSARSA",
    "GAAC",
    "GPUCB",
    "MPO",
    "PPO",
    "QLearning",
    "REINFORCE",
    "REPS",
    "Random",
    "SAC",
    "SARSA",
    "SVG0",
    "SoftQLearning",
    "TD3",
    "TRPO",
    "VMPO",
]

MODEL_BASED = ["BPTT", "MPC", "SVG"]
MODEL_AUGMENTED = ["Dyna", "MVE", "STEVE"]
DATA_AUGMENTED = ["DataAugmentation"]
AGENTS = MODEL_FREE + MODEL_BASED + MODEL_AUGMENTED + DATA_AUGMENTED
