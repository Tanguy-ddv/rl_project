from stable_baselines3 import PPO
import numpy as np

def performPPO(env, agent: PPO, render: bool = False
			) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], float, int]:
	"""
	Perform one episode for the agent on the env.
	Store the outcome at each step.

	Parameters:
	----
	env: The gym environment
	agent: the PPO agent to perform one episode.
	render: if True, show the agent performing on the env.

	Returns:
	----
	previous_states: A list of np.ndarray being the list of previous states during the episode
	actions: A list of np.ndarray being the list of actions performed during the episode.
	states: A list of np.ndarray being the list of states of the episode.
	cumulated_reward: The cumulated reward.
	episode_length: The length of the episode.
	"""
	
    # Reset the training data
	done = False
	cumulated_reward = 0
	episode_length = 0
	actions = []
	previous_states = []
	states = []

	state = env.reset()

	while not done:  # Loop until the episode is over

		action, _ = agent.predict(observation=state, deterministic=True)
		previous_state = state

		state, reward, done, _info = env.step(action)
		if render:
			env.render()

		cumulated_reward += reward
		states.append(state)
		actions.append(action)
		previous_states.append(previous_state)

		episode_length +=1

	return previous_states, actions, states, cumulated_reward, episode_length
