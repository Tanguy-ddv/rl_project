"""
Train an RL agent on the OpenAI Gym Hopper environment using
REINFORCE and Actor-critic algorithms
"""

import torch

from agent.actor_critic_agent import Agent, ActorCriticAgent

_RED = "\033[1;31;40m"
_GREEN = "\033[1;32;40m"
_YELLOW = "\033[1;33;40m"

def train(
	env,
	agent: Agent,
	n_episode:int = 10e3,
	verbose: int = 1,
	early_stopping_threshold: int = None
):
	
	"""
	Train the agent
	
	Parameters:
	----
	env: The gym environment
	agent: An actorcritic or Reinfocement agent to be trained.
	n_episode: The number of episode for the training.
	verbose: The amount of verbose asked during the training.
	early_stopping_threshold: If the system don't improve himself during this number of episode, stop the training
	
	Returns:
	----
	rewards: All the rewards of all the steps of each episodes 
	episode_lengths: The number of steps of every episodes
	final_rewards: The cumulated reward of all the episodes, corresponds to [sum(R) for R in rewards]
	states: The history of all the states of each step of each episode
	max_train_reward: The bets cumulated reward achieve during the training.
	"""
 
	PRINT_EVERY = int(n_episode/(verbose*5)) if verbose else 0

	# Initilisation for the data measurement.
	rewards = []
	final_rewards = []
	episode_lengths = []
	states = []

	max_train_reward = None

	# Only for printing the evolution.
	if verbose:
		last_episode_lengths = []
		last_rewards = []
		previous_train_reward = 0
	
	last_best_model_delta = 0		

	# Start of the training loop
	for episode in range(n_episode):
		state = env.reset()  # Reset the environment and observe the initial state

		# Reset the training data
		done = False
		train_reward = 0
		episode_length = 0
		rewards.append([])
		states.append([])
		last_best_model_delta +=1

		while not done:  # Loop until the episode is over

			action, action_probabilities = agent.get_action(state)
			previous_state = state

			state, reward, done, _info = env.step(action.detach().cpu().numpy())
			agent.store_outcome(previous_state, state, action_probabilities, reward, done)

			train_reward += reward
			rewards[-1].append(train_reward)
			states[-1].append(list(state))

			episode_length +=1

		# Update the agent at the end of the episode
		agent.update_policy()
		agent.clear_history()

		# Update the episodic statistics.
		episode_lengths.append(episode_length)
		final_rewards.append(train_reward)
		if verbose:
			last_episode_lengths.append(episode_length)
			last_rewards.append(train_reward)

		# Save the best model
		if max_train_reward is None or train_reward > max_train_reward:
			max_train_reward = train_reward
			torch.save(agent.policy.state_dict(), "best_model.mdl")
			with open("episode_of_best.txt", 'w') as f:
				f.writable(str(episode))
			if isinstance(agent, ActorCriticAgent):
				torch.save(agent.critic.state_dict(), "best_critic.mdl")
			if verbose:
				color = _GREEN
			last_best_model_delta = 0

		# Print the average episode length and reward of the last episodes
		if PRINT_EVERY and episode and (episode)%PRINT_EVERY == 0:
			train_reward = sum(last_rewards)/len(last_rewards)
			episode_length = sum(last_episode_lengths)/len(last_episode_lengths)

			if color == _RED and train_reward > previous_train_reward:
				color = _YELLOW
			print(f"{color}Episode: {episode} | Average return: {train_reward:.2f} | Average episode length: {episode_length:.2f}\033[0m")

			color = _RED
			previous_train_reward = train_reward
			last_rewards.clear()
			last_episode_lengths.clear()
		
		# Early stop if the last trained model has been caught too episodes ago
		if early_stopping_threshold and last_best_model_delta > early_stopping_threshold:
			break
	
	return rewards, episode_lengths, final_rewards, states, max_train_reward