import matplotlib.pyplot as plt
import numpy as np
from rllib.util.rollout import rollout_agent


def rollout(environment, agent, num_episodes, max_steps, test_episodes=1):
    rollout_agent(environment, agent, num_episodes=num_episodes, max_steps=max_steps)

    for key, log in agent.logs.items():
        plt.plot(log.episode_log)
        plt.xlabel('Episode')
        plt.ylabel(' '.join(key.split('_')).capitalize())
        plt.title('{} in {}'.format(agent.name, environment.name))
        plt.show()
    print(repr(agent))
    rollout_agent(environment, agent, max_steps=max_steps, num_episodes=test_episodes,
                  render=True)
    print('Test Rewards:',
          np.array(agent.logs['rewards'].episode_log[-test_episodes]).mean()
          )

