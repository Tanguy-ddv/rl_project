"""Test an RL agent on the OpenAI Gym Hopper environment"""

from agent.agent import Agent

def test(
	env,
	agent: Agent,
	n_episodes: int=10,
	render: bool= True
):
	rewards = []
	episode_lengths = []
	final_rewards = []
	states = []
	max_train_reward = 0
	for episode in range(n_episodes):

		done = False
		test_reward = 0
		state = env.reset()
		rewards.append([])
		states.append([])

		while not done:

			action, _ = agent.get_action(state, evaluation=True)
			state, reward, done, info = env.step(action.detach().cpu().numpy())

			if render:
				env.render()

			test_reward += reward
			rewards[-1].append(reward)
			states[-1].append(list(state))
		
		if max_train_reward is None or test_reward > max_train_reward:
			max_train_reward = test_reward

		final_rewards.append(test_reward)
		episode_lengths.append(len(rewards[-1]))
		print(f"Episode: {episode} | Return: {test_reward:.2f}")
	
	return rewards, episode_lengths, final_rewards, states, max_train_reward