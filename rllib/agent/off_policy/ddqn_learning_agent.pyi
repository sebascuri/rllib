from rllib.agent import QLearningAgent
from rllib.algorithms.q_learning import DDQN


class DDQNAgent(QLearningAgent):
    algorithm: DDQN

