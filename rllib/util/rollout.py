"""Helper functions to conduct a rollout with policies or agents."""

import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from tqdm import tqdm

from rllib.dataset.datatypes import Observation
from rllib.util.training.utilities import Evaluate
from rllib.util.utilities import get_entropy_and_log_p, tensor_to_distribution


def step_env(environment, state, action, action_scale, pi=None, render=False):
    """Perform a single step in an environment."""
    try:
        next_state, reward, done, info = environment.step(action)
    except TypeError:
        next_state, reward, done, info = environment.step(action.item())

    if not isinstance(action, torch.Tensor):
        action = torch.tensor(action, dtype=torch.get_default_dtype())

    if pi is not None:
        try:
            with torch.no_grad():
                entropy, log_prob_action = get_entropy_and_log_p(
                    pi, action, action_scale
                )
        except RuntimeError:
            entropy, log_prob_action = 0.0, 1.0
    else:
        entropy, log_prob_action = 0.0, 1.0

    observation = Observation(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=done,
        entropy=entropy,
        log_prob_action=log_prob_action,
    ).to_torch()
    state = next_state
    if render:
        environment.render()
    return observation, state, done, info


def step_model(
    dynamical_model,
    reward_model,
    termination_model,
    state,
    action,
    done=None,
    action_scale=1.0,
    pi=None,
):
    """Perform a single step in an dynamical model."""
    # Sample a next state
    next_state_out = dynamical_model(state, action)
    next_state_distribution = tensor_to_distribution(next_state_out)

    if next_state_distribution.has_rsample:
        next_state = next_state_distribution.rsample()
    else:
        next_state = next_state_distribution.sample()

    # Sample a reward
    reward_distribution = tensor_to_distribution(
        reward_model(state, action, next_state)
    )
    if reward_distribution.has_rsample:
        reward = reward_distribution.rsample().squeeze(-1)
    else:
        reward = reward_distribution.sample().squeeze(-1)
    if done is None:
        done = torch.zeros_like(reward).bool()
    reward *= (~done).float()

    # Check for termination.
    if termination_model is not None:
        done = done + (  # "+" is a boolean "or".
            tensor_to_distribution(termination_model(state, action, next_state))
            .sample()
            .bool()
        )

    if pi is not None:
        try:
            entropy, log_prob_action = get_entropy_and_log_p(pi, action, action_scale)
        except RuntimeError:
            entropy, log_prob_action = 0.0, 1.0
    else:
        entropy, log_prob_action = 0.0, 1.0

    observation = Observation(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=done.float(),
        entropy=entropy,
        log_prob_action=log_prob_action,
        next_state_scale_tril=next_state_out[-1],
    ).to_torch()

    # Update state.
    next_state = torch.zeros_like(state)
    next_state[~done] = observation.next_state[~done]  # update next state.
    next_state[done] = state[done]  # don't update next state.

    return observation, next_state, done


def record(environment, agent, path, num_episodes=1, max_steps=1000):
    """Record an episode."""
    recorder = VideoRecorder(environment, path=path)
    for _ in range(num_episodes):
        state = environment.reset()
        agent.set_goal(environment.goal)

        done = False
        time_step = 0
        while not done:
            action = agent.act(state)
            observation, state, done, info = step_env(
                environment, state, action, agent.policy.action_scale
            )
            recorder.capture_frame()

            time_step += 1
            if max_steps <= time_step:
                break

    recorder.close()


def rollout_episode(environment, agent, max_steps, render):
    """Rollout a full episode."""
    state = environment.reset()
    agent.set_goal(environment.goal)
    agent.start_episode()
    done = False
    time_step = 0
    while not done:
        action = agent.act(state)
        obs, state, done, info = step_env(
            environment=environment,
            state=state,
            action=action,
            action_scale=agent.policy.action_scale,
            pi=agent.pi,
            render=render,
        )
        agent.observe(obs)
        # Log info.
        agent.logger.update(**info)

        time_step += 1
        if max_steps <= time_step:
            break

    agent.end_episode()


def rollout_agent(
    environment,
    agent,
    num_episodes=1,
    max_steps=1000,
    render=False,
    print_frequency=0,
    callback_frequency=0,
    eval_frequency=0,
    save_milestones=None,
    callbacks=None,
):
    """Conduct a rollout of an agent in an environment.

    Parameters
    ----------
    environment: AbstractEnvironment
        Environment with which the abstract interacts.
    agent: AbstractAgent
        Agent that interacts with the environment.
    num_episodes: int, optional (default=1)
        Number of episodes.
    max_steps: int.
        Maximum number of steps per episode.
    render: bool.
        Flag that indicates whether to render the environment or not.
    print_frequency: int, optional.
        Print agent stats every `print_frequency' episodes if > 0.
    callback_frequency: int, optional.
        Plot agent callbacks every `plot_frequency' episodes if > 0.
    eval_frequency: int, optional.
        Evaluate agent every 'eval_frequency' episodes if > 0.
    save_milestones: List[int], optional.
        List with episodes in which to save the agent.
    callbacks: List[Callable[[AbstractAgent, AbstractEnvironment,int], None]], optional.
        List of functions for evaluating/plotting the agent.
    """
    save_milestones = list() if save_milestones is None else save_milestones
    callbacks = list() if callbacks is None else callbacks
    for episode in tqdm(range(num_episodes)):
        rollout_episode(environment, agent, max_steps, render)

        if print_frequency and episode % print_frequency == 0:
            print(agent)
        if callback_frequency and episode % callback_frequency == 0:
            for plot_callback in callbacks:
                plot_callback(agent, environment, episode)

        if episode in save_milestones:
            agent.save(f"{agent.name}_{episode}.pkl")

        if eval_frequency and episode % eval_frequency == 0:
            with Evaluate(agent):
                rollout_episode(environment, agent, max_steps, render)
    agent.end_interaction()


def rollout_policy(environment, policy, num_episodes=1, max_steps=1000, render=False):
    """Conduct a rollout of a policy in an environment.

    Parameters
    ----------
    environment: AbstractEnvironment
        Environment with which the policy interacts.
    policy: AbstractPolicy
        Policy that interacts with the environment.
    num_episodes: int, optional (default=1)
        Number of episodes.
    max_steps: int.
        Maximum number of steps per episode.
    render: bool.
        Flag that indicates whether to render the environment or not.

    Returns
    -------
    trajectories: List[Trajectory]=List[List[Observation]]
        A list of trajectories.

    """
    trajectories = []
    for _ in tqdm(range(num_episodes)):
        state = environment.reset()
        done = False
        trajectory = []
        with torch.no_grad():
            time_step = 0
            while not done:
                pi = tensor_to_distribution(
                    policy(torch.tensor(state, dtype=torch.get_default_dtype())),
                    **policy.dist_params,
                )
                action = pi.sample()
                if not policy.discrete_action:
                    action = policy.action_scale * action.clamp_(-1.0, 1.0)
                obs, state, done, info = step_env(
                    environment=environment,
                    state=state,
                    action=action.detach().numpy(),
                    action_scale=policy.action_scale,
                    pi=pi,
                    render=render,
                )
                trajectory.append(obs)

                time_step += 1
                if max_steps <= time_step:
                    break

        trajectories.append(trajectory)
    return trajectories


def rollout_model(
    dynamical_model,
    reward_model,
    policy,
    initial_state,
    initial_action=None,
    termination_model=None,
    max_steps=1000,
):
    """Conduct a rollout of a policy interacting with a model.

    Parameters
    ----------
    dynamical_model: AbstractModel
        Dynamical Model with which the policy interacts.
    reward_model: AbstractModel.
        Reward Model with which the policy interacts.
    policy: AbstractPolicy
        Policy that interacts with the environment.
    initial_state: State
        Starting states for the interaction.
    initial_action: Action.
        Starting action for the interaction.
    termination_model: AbstractModel.
        Termination condition to finish the rollout.
    max_steps: int.
        Maximum number of steps per episode.

    Returns
    -------
    trajectory: Trajectory=List[Observation]
        A list of observations.

    Notes
    -----
    It will try to do the re-parametrization trick with the policy and models.

    TODO: Parallelize it!.
    """
    trajectory = list()
    state = initial_state
    done = torch.full(state.shape[:-1], False, dtype=torch.bool)

    assert max_steps > 0
    for i in range(max_steps):
        if policy is not None:
            pi = tensor_to_distribution(policy(state), **policy.dist_params)
            action_scale = policy.action_scale
        else:
            assert max_steps == 1
            pi, action_scale = None, 1.0

        if i == 0 and initial_action is not None:
            action = initial_action
        else:
            # Sample an action
            if pi.has_rsample:
                action = pi.rsample()
            else:
                action = pi.sample()
            if not policy.discrete_action:
                action = policy.action_scale * action.clamp_(-1.0, 1.0)

        observation, next_state, done = step_model(
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            termination_model=termination_model,
            state=state,
            action=action,
            action_scale=action_scale,
            done=done,
            pi=pi,
        )
        trajectory.append(observation)

        state = next_state
        if torch.all(done):
            break

    return trajectory


def rollout_actions(
    dynamical_model,
    reward_model,
    action_sequence,
    initial_state,
    termination_model=None,
):
    """Conduct a rollout of an action sequence interacting with a model.

    Parameters
    ----------
    dynamical_model: AbstractModel
        Dynamical Model with which the policy interacts.
    reward_model: AbstractReward, optional.
        Reward Model with which the policy interacts.
    action_sequence: Action
        Action Sequence that interacts with the environment.
        The dimensions are [horizon x num_samples x dim_action].
    initial_state: State
        Starting states for the interaction.
        The dimensions are [1 x num_samples x dim_state].
    termination_model: Callable.
        Termination condition to finish the rollout.

    Returns
    -------
    trajectory: Trajectory=List[Observation]
        A list of observations.

    Notes
    -----
    It will try to do the re-parametrization trick with the policy and models.
    """
    trajectory = list()
    state = initial_state
    done = torch.full(state.shape[:-1], False, dtype=torch.bool)

    for action in action_sequence:  # Normalized actions

        observation, next_state, done = step_model(
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            termination_model=termination_model,
            state=state,
            action=action,
            action_scale=1.0,
            done=done,
        )
        trajectory.append(observation)

        state = next_state
        if torch.all(done):
            break

    return trajectory
