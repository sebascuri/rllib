from .abstract_agent import AbstractAgent
from .bandit import GPUCBAgent
from .model_based import MBDPGAgent, MBMPPOAgent, MBSACAgent, MPCAgent
from .off_policy import (
    DDQNAgent,
    DPGAgent,
    DQNAgent,
    MPPOAgent,
    QLearningAgent,
    QREPSAgent,
    REPSAgent,
    SACAgent,
    SoftQLearningAgent,
    TD3Agent,
)
from .on_policy import (
    A2CAgent,
    ActorCriticAgent,
    ExpectedActorCriticAgent,
    ExpectedSARSAAgent,
    GAACAgent,
    REINFORCEAgent,
    SARSAAgent,
)
from .random_agent import RandomAgent

AGENTS = [
    "GPUCB",
    "MBDPG",
    "MBMPPO",
    "MBSAC",
    "MPC",
    "DDQN",
    "DPG",
    "DQN",
    "MPPO",
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
]
