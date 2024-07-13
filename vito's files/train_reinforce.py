"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse

import matplotlib.pyplot as plt
import torch
import gym

from env.custom_hopper import *
from agent_reinforce import Agent, Policy
from timeit import default_timer as timer
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default= 10000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=100, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()




def main():

	start_time = timer()

	env = gym.make('CustomHopper-source-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())


	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	agent = Agent(policy, device=args.device)

	rewards_per_episode = [] #to store rewards for the plot
	
	"""
	#to load the already trained algorithm
	if os.path.exists("model_reinforce.mdl"):
		agent.policy.load_state_dict(torch.load("model_reinforce.mdl"))
	"""

	for episode in range(args.n_episodes):
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state

		while not done:  # Loop until the episode is over

			action, action_probabilities = agent.get_action(state)
			previous_state = state
			
			state, reward, done, info = env.step(action.detach().cpu().numpy())

			agent.store_outcome(previous_state, state, action_probabilities, reward, done)
			train_reward += reward
		
		agent.update_policy_reinforce()
		
		rewards_per_episode.append(train_reward)
		
		#to save best policy
		if (train_reward == max(rewards_per_episode)):
			best_reward = train_reward
			best_policy = agent.policy


		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode+1)
			print('Episode return:', train_reward)
			#rewards_per_episode.append(train_reward)

	torch.save(best_policy.state_dict(), "model_reinforce.mdl")		
	
	end_time = timer()
	print(f"Training time: {end_time-start_time:.3f} seconds")

	print(f"The best policy has reward equal to {best_reward} ")


	with open('training_output.txt', 'w') as f:
		f.write("Episode\tReward\n")
		for i, reward in enumerate(rewards_per_episode):
			if (i+1)%args.print_every == 0:
				f.write(f"{int(i)}\t{reward}\n")
	
	plt.plot(np.arange(1, len(rewards_per_episode)+1), rewards_per_episode)
	plt.xlabel('Episode')
	plt.ylabel('Reward')
	plt.title('Rewards x episodes')
	plt.grid(True)
	plt.savefig('training_rewards_plot.png')
	plt.show()


if __name__ == '__main__':
	main()