from .abstract_agent import AbstractAgent
from .bandit import GPUCBAgent
from .model_based import MBDPGAgent, MBMPOAgent, MBSACAgent, MPCAgent
from .off_policy import (
    DDQNAgent,
    DPGAgent,
    DQNAgent,
    MPOAgent,
    QLearningAgent,
    QREPSAgent,
    REPSAgent,
    SACAgent,
    SoftQLearningAgent,
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

AGENTS = [
    "GPUCB",
    "MBDPG",
    "MBMPO",
    "MBSAC",
    "MPC",
    "DDQN",
    "DPG",
    "DQN",
    "MPO",
    "QLearning",
    "QREPS",
    "REPS",
    "SAC",
    "SoftQLearning",
    "TD3",
    "A2C",
    "ActorCritic",
    "ExpectedActorCritic",
    "ExpectedSARSA",
    "GAAC",
    "REINFORCE",
    "SARSA",
    "PPO",
    "TRPO",
    "Random",
    "VMPO",
]
