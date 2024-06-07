"""
The perform function is used to perform one episode of the agent on the given environment.
"""

import numpy as np

from agent.actor_critic_agent import Agent

def perform(env, agent: Agent, store: bool = True, render: bool = False) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], float, int]:
	"""
	Perform one episode for the agent on the env.
	Store the outcome at each step.

	Parameters:
	----
	env: The gym environment
	agent: the REINFORCE or Actor-Critic agent to train one episode.
	initial_state: the state of the env at the very beginning, should be obtain previously via env.reset()
	store: if True, store the episode outcome on the agent history, else don't.
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

		action, action_probabilities = agent.get_action(state)
		previous_state = state

		previous_state = state

		state, reward, done, _info = env.step(action.detach().cpu().numpy())
		if store:
			agent.store_outcome(previous_state, state, action_probabilities, reward, done)
		if render:
			env.render()

		cumulated_reward += reward
		states.append(state)
		actions.append(action.numpy())
		previous_states.append(previous_state)

		episode_length +=1

	return previous_states, actions, states, cumulated_reward, episode_length