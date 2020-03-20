import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from rllib.util.rollout import rollout_agent


def _model_loss(model, state, action, next_state):
    mean, cov = model(state, action)
    y_pred = mean
    y = next_state
    if torch.all(cov == 0):
        loss = torch.mean((y_pred - y) ** 2)
    else:
        loss = ((mean - y) @ torch.inverse(cov) @ (mean - y).T).mean()
        loss += torch.mean(torch.logdet(cov))
    return loss


def train_model(model, data, num_iter, optimizer):
    num_data = data.state.shape[0]
    for _ in range(num_iter):
        optimizer.zero_grad()

        loss = _model_loss(model, data.state[num_data // 2:],
                           data.action[num_data // 2:],
                           data.next_state[num_data // 2:])
        loss.backward()
        optimizer.step()
        print('train:', loss.item())

        with torch.no_grad():
            loss = _model_loss(model, data.state[:num_data // 2],
                               data.action[:num_data // 2],
                               data.next_state[:num_data // 2])
            print('test:', loss.item())


def train_agent(agent, environment, num_episodes, max_steps):
    """Train an agent in an environment.

    Parameters
    ----------
    agent: AbstractAgent
    environment: AbstractEnvironment
    num_episodes: int
    max_steps: int
    """
    agent.train()
    rollout_agent(environment, agent, num_episodes=num_episodes, max_steps=max_steps)

    for key, log in agent.logs.items():
        plt.plot(log.episode_log)
        plt.xlabel('Episode')
        plt.ylabel(' '.join(key.split('_')).capitalize())
        plt.title('{} in {}'.format(agent.name, environment.name))
        plt.show()
    print(repr(agent))


def evaluate_agent(agent, environment, num_episodes, max_steps):
    """Evaluate an agent in an environment.

    Parameters
    ----------
    agent: AbstractAgent
    environment: AbstractEnvironment
    num_episodes: int
    max_steps: int
    """
    agent.eval()
    rollout_agent(environment, agent, max_steps=max_steps, num_episodes=num_episodes,
                  render=True)
    print('Test Rewards:',
          np.array(agent.logs['rewards'].episode_log[-num_episodes]).mean()
          )


def save_agent(agent, file_name):
    """Save the agent as a pickled file.

    Parameters
    ----------
    agent: AbstractAgent.
    file_name: str.

    """
    with open(file_name, 'wb') as file:
        pickle.dump(agent, file)


def load_agent(file_name):
    """Load and return the agent at a given file location.

    Parameters
    ----------
    file_name: str.

    Returns
    -------
    agent: AbstractAgent.
    """
    with open(file_name, 'rb') as f:
        agent = pickle.load(f)

    return agent
