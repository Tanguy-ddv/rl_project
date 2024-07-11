import argparse

import matplotlib.pyplot as plt
import torch
import pandas as pd
import gym

from env.custom_randomized_hopper import *
from agent_actor_critic_lstm import Agent, Actor, Critic

from timeit import default_timer as timer
import os
import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default= 5000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=10, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()

seed = 17

def main():
	start_time = timer()
	
	
	env = gym.make('CustomHopper-source-v0')

	env.seed(seed)
	#np.random.seed(seed)        
	#torch.manual_seed(seed)   


	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	actor = Actor(observation_space_dim, action_space_dim, 64)
	critic = Critic(observation_space_dim, action_space_dim, 64)
	agent = Agent(actor, critic, device=args.device)    

	# Management of the directories
	model_dir = 'models_ac_lstm'
	seed_dir = os.path.join(model_dir, f'ac_seed:{seed}')
	os.makedirs(seed_dir, exist_ok=True)
	model_file_pattern = os.path.join(seed_dir, f'model_actor_critic_seed:{seed}_nEp:*.mdl')

	# Get the most recent model if exists
	existing_models = glob.glob(model_file_pattern)
	if existing_models:
		existing_models.sort(key=lambda x: int(x.split('_nEp:')[-1].split('.mdl')[0]))
		latest_model = existing_models[-1]
		agent.actor.load_state_dict(torch.load(latest_model))
		start_episode = int(latest_model.split('_nEp:')[-1].split('.mdl')[0])
	else:
		start_episode = 0

	results_file = os.path.join(seed_dir, f'results_training_seed:{seed}.txt')

	# Create file if it does not exist
	if not os.path.exists(results_file):
		with open(results_file, 'w') as f:
			f.write("Episode\tReward\n")
	

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())

	rewards_per_episode = [] 


	for episode in range(start_episode, start_episode + args.n_episodes):
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state
		hidden_state = None
		while not done:  # Loop until the episode is over

			action, action_probabilities, hidden_state = agent.get_action(state, hidden_state)
			previous_state = state
			state, reward, done, info = env.step(action.detach().cpu().numpy())
			agent.store_outcome(previous_state, state, action_probabilities, reward, done)
			train_reward += reward

		agent.update_policy_actor_critic()

		
		# Append results to the file
		with open(results_file, 'a') as f:
			f.write(f"{episode+1}\t{train_reward:.2f}\n")
		
		rewards_per_episode.append(train_reward)
		
		#to save best policy
		if (train_reward == max(rewards_per_episode)):
			best_reward = train_reward
			best_policy = agent.actor

		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode+1)
			print('Episode return:', train_reward)
			#rewards_per_episode.append(train_reward)

	model_save_path = os.path.join(seed_dir, f'model_actor_critic_seed:{seed}_nEp:{episode+1}.mdl')
	torch.save(agent.actor.state_dict(), model_save_path)

	model_best_policy_path = os.path.join(seed_dir, f'model_actor_critic_best_policy_seed:{seed}_nEp:{episode+1}.mdl')

	end_time = timer()
	print(f"Training time: {end_time-start_time:.3f} seconds")

	print(f"The best policy has reward equal to {best_reward} ")

	# Read rewards from file for plotting
	results_df = pd.read_csv(results_file, sep='\t')

	plt.plot(results_df['Episode'], results_df['Reward'])
	plt.xlabel('Episode')
	plt.ylabel('Reward')
	plt.title('Rewards x episodes Actor Critic with LSTM')
	plt.grid(True)
	plot_save_path = os.path.join(seed_dir, f'plot_rewards_seed:{seed}_nEp_{start_episode + args.n_episodes}.png')
	plt.savefig(plot_save_path)
	plt.show()


if __name__ == '__main__':
	main()