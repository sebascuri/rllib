from rllib.environment.systems import InvertedPendulum
from rllib.environment import SystemEnvironment
from rllib.policy import NNPolicy
from rllib.value_function import NNValueFunction
import torch.nn as nn
import torch.optim as optim
from rllib.model import LinearModel
import numpy as np
from rllib.agent import DDPGAgent


system = InvertedPendulum(mass=0.1, length=0.5, friction=0.)
system = system.linearize()
q = np.eye(2)
r = 0.01 * np.eye(1)
gamma = 0.99


environment = SystemEnvironment(system, initial_state=None, termination=None, reward=None)

model = LinearModel(system.a, system.b)
policy = NNPolicy(dim_state=system.dim_state, dim_action=system.dim_action,
                  layers=[], biased_head=False, deterministic=True)  # Linear policy.
value_function = NNValueFunction(dim_state=system.dim_state, layers=[64, 64],
                                 biased_head=False)

loss_function = nn.MSELoss()
value_optimizer = optim.Adam(value_function.parameters, lr=5e-4)
policy_optimizer = optim.Adam(policy.parameters, lr=5e-4)

agent = DDPGAgent(value_function, policy, None, loss_function, value_optimizer,
                  policy_optimizer, None)