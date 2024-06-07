"""
Test an Agent on the environment for n episodes.
"""

import numpy as np
from ..actor_critic_agent import Agent
from .perform import perform

def test(
	env,
	agent: Agent,
	n_episodes: int=10,
	render: bool = True,
	seed: int = None,
):
	if seed is not None:
		env.seed(seed)
	rewards = []
	episode_lengths = []
	max_test_reward = None

	for _ in range(n_episodes):

		_, _, _, test_reward, episode_length, _ = perform(env, agent, False, render)

		if max_test_reward is None or test_reward > max_test_reward:
			max_test_reward = test_reward

		rewards.append(test_reward)
		episode_lengths.append(episode_length)
	
	print(f"Mean return: {np.mean(rewards):.2f} | Std return: {np.std(rewards):.2f} | Max return: {max_test_reward:.2f} | Mean episode length: {np.mean(episode_lengths)}")
	
	return rewards, episode_lengths, max_test_reward
