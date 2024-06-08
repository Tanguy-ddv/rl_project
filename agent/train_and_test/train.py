"""
Train an RL agent on the OpenAI Gym Hopper environment using
REINFORCE and Actor-critic algorithms
"""

import torch

from ..actor_critic_agent import Agent, ActorCriticAgent
from .perform import perform

_RED = "\033[1;31;40m"
_GREEN = "\033[1;32;40m"
_YELLOW = "\033[1;33;40m"

def train(
	env,
	agent: Agent,
	n_episode: int = 10e3,
	verbose: int = 1,
	early_stopping_threshold: int = None,
	save_output: str = "",
	episode0: int = 0,
):
	"""
	Train the agent
	
	Parameters:
	----
	env: The gym environment
	agent: An actor-Critic or REINFORCE agent to be trained.
	n_episode: The number of episode for the training.
	verbose: The amount of verbose asked during the training.
	early_stopping_threshold: If the system don't improve himself during this number of episode, stop the training
	save_output: str, the path to the folder where the best and the last model. must end with '/' or be '' for saving in the current folder.
	episode0: Only for fractionned call of the function with verbose, to not restart the episode count.
	
	Returns:
	----
	episode_lengths: The number of steps of every episodes
	rewards: The cumulated reward of all the episodes, corresponds to [sum(R) for R in rewards]
	max_train_reward: The bets cumulated reward achieve during the training.
	best_model_episode: The episode of the best model.
	"""
	PRINT_EVERY = 1000//verbose if verbose else 0

	# Initilisation for the data measurement.
	rewards = []
	episode_lengths = []
	max_train_reward = None

	# Only for printing the evolution.
	previous_train_reward = 0
	
	# The episode of the last best model.
	best_model_episode = 0		

	# Start of the training loop
	for episode in range(n_episode):
		# Train one episode
		_, _, _, train_reward, episode_length, _ = perform(env, agent, True)

		# Update the agent at the end of the episode
		agent.update_policy()
		agent.clear_history()

		# Update the episodic statistics.
		episode_lengths.append(episode_length)
		rewards.append(train_reward)

		# Save the best model
		if max_train_reward is None or train_reward > max_train_reward:
			max_train_reward = train_reward
			torch.save(agent.policy.state_dict(), save_output + "best_model.mdl")
			if isinstance(agent, ActorCriticAgent):
				torch.save(agent.critic.state_dict(), save_output + "best_critic.mdl")
			if verbose:
				color = _GREEN
			best_model_episode = episode

		# Print the average episode length and reward of the last episodes
		if PRINT_EVERY and episode and (episode)%PRINT_EVERY == 0:
			last_train_reward = sum(rewards[-PRINT_EVERY:])/PRINT_EVERY
			episode_length = sum(episode_lengths[-PRINT_EVERY:])/PRINT_EVERY

			if color == _RED and train_reward > previous_train_reward:
				color = _YELLOW
			print(f"{color}Episode: {episode+episode0} | Average return: {last_train_reward:.2f} | Average episode length: {episode_length:.2f}\033[0m")

			color = _RED
			previous_train_reward = last_train_reward
		
		# Early stop if the last trained model has been caught too episodes ago
		if early_stopping_threshold and episode - best_model_episode > early_stopping_threshold:
			break

		torch.save(agent.policy.state_dict(), save_output + "model.mdl")
		if isinstance(agent, ActorCriticAgent):
			torch.save(agent.critic.state_dict(), save_output + "critic.mdl")

	
	return episode_lengths, rewards, max_train_reward, best_model_episode