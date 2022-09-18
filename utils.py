from collections import deque
from typing import List, Tuple
from agent import Agent
from tqdm import tqdm
import numpy as np
from unityagents import UnityEnvironment
import logging
from matplotlib import pyplot as plt


def ddpg_train(agent: Agent, env: UnityEnvironment, n_episodes=2000, max_t=1000, print_every=100, min_mean_score=30) -> List[float]:
    """
    Train the agent using DDPG algorithm with the given environment.

    :param agent: Agent to train,
    :param env: Unity environment,
    :param n_episodes: Number of episodes to train the agent,
    :param max_t: Maximum number of time steps per episode
    :param print_every: Print the average score every print_every episode
    :param min_mean_score: Minimum mean score to stop the training

    :return: List of scores
    """

    # Extract the default brain
    brain_name = env.brain_names[0]

    scores = []
    scores_window = deque(maxlen=100)

    # Loop over episodes
    for episode in tqdm(range(n_episodes)):
        env_info = env.reset(train_mode=True)[brain_name]            # reset the environment
        num_agents = len(env_info.agents)                            # get the number of agents (constant for all episodes)
        states = env_info.vector_observations                        # get the current state (for each agent)
        agent.reset()                                                # reset the agent noise
        score = np.zeros(num_agents)                                 # initialize the score (for each agent)

        # Loop over time steps, until the episode is done or max_t is reached
        for t in range(max_t):
            actions = agent.act(states)                              # select an action (for each agent)

            env_info = env.step(actions)[brain_name]                 # send the action to the environment
            next_states = env_info.vector_observations               # get the next states (for each agent)
            rewards = env_info.rewards                               # get the rewards (for each agent)
            dones = env_info.local_done                              # see if episodes have finished (for each agent)

            agent.step(states, actions, rewards, next_states, dones) # update the agents
            score += rewards                                         # update the scores (for each agent)
            states = next_states                                     # roll over the state to next time step (for each agent)

            if np.any(dones):                                      # exit loop if episode finished for any agent
                break

        agent.checkpoint()                                           # save the model weights

        scores.append(np.mean(score))                                # save the average score
        scores_window.append(np.mean(score))                         # save the average score for the last 100 episodes

        if episode % print_every == 0:
            logging.info('\rEpisode: \t{} \tScore: \t{:.2f} \tAverage Score: \t{:.2f}'.format(episode, np.mean(score), np.mean(scores_window)))

        if np.mean(scores_window) >= min_mean_score:
            logging.info('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
            break

    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    return scores
