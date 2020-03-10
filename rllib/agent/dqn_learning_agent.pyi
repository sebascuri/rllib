from rllib.agent import QLearningAgent
from rllib.algorithms.q_learning import DQN


class DQNAgent(QLearningAgent):
    q_learning: DQN

