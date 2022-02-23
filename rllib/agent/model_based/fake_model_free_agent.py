"""Python Script Template."""
from importlib import import_module

from rllib.environment.fake_environment import FakeEnvironment
from rllib.util.neural_networks.utilities import DisableGradient
from rllib.util.training.agent_training import train_agent

from .model_based_agent import ModelBasedAgent


class FakeModelFreeAgent(ModelBasedAgent):
    """A Fake model-free agent is a agent that is optimized on a learned model.

    Parameters
    ----------
    model_free_agent: AbstractAgent.
        Agent to run interacting with the model.
    fake_episodes: int.
        How many fake episodes (per `num_rollouts`) to rollout the agent.
    fake_horizon: int.
        How long should fake episodes be.

    """

    def __init__(
        self,
        model_free_agent,
        fake_episodes=100,
        fake_horizon=10,
        num_rollouts=1,
        train_frequency=0,
        *args,
        **kwargs,
    ):
        super().__init__(
            num_rollouts=num_rollouts, train_frequency=train_frequency, *args, **kwargs
        )
        self.model_free_agent = model_free_agent
        self.model_free_agent.logger.change_log_dir(f"{self.logger.log_dir}/base_agent")

        self.fake_episodes = fake_episodes
        self.fake_horizon = fake_horizon
        self.policy = self.model_free_agent.policy

    def learn(self, memory=None):
        """Learn the model-free agent by rolling it out on the fake environment."""
        #

        def sample_initial_states():
            """Get initial states to sample from."""
            # Samples from experience replay empirical distribution.
            obs, *_ = self.memory.sample_batch(1)
            for transform in self.memory.transformations:
                obs = transform.inverse(obs)
            initial_states = obs.state[:, 0, :]  # obs is an n-step return.
            return initial_states.squeeze(0)

        fake_env = FakeEnvironment(
            dynamical_model=self.dynamical_model,
            reward_model=self.reward_model,
            termination_model=self.termination_model,
            initial_state_fn=sample_initial_states,
        )
        with DisableGradient(
            self.dynamical_model, self.reward_model, self.termination_model
        ):
            train_agent(
                agent=self.model_free_agent,
                environment=fake_env,
                plot_flag=False,
                max_steps=self.fake_horizon,
                num_episodes=self.fake_episodes,
            )

    @classmethod
    def default(cls, environment, base_agent="SAC", *args, **kwargs):
        """Create default agent."""
        model_free_agent = getattr(
            import_module("rllib.agent"), f"{base_agent}Agent"
        ).default(environment, *args, **kwargs)

        return super().default(
            environment, model_free_agent=model_free_agent, *args, **kwargs
        )

    def load(self, path):
        """Load agent from path."""
        super().load(path)
        self.model_free_agent.load(f"{path}/base_agent/")
