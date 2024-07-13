from env.custom_randomized_hopper import *

import os
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from env.custom_randomized_hopper import CustomHopper


# Define ADR network
class ADRNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ADRNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softplus()  # To ensure positive scaling factors
        )
    
    def forward(self, x):
        return self.fc(x)


# ADR Wrapper
class ADRWrapper(gym.Wrapper):
    def __init__(self, env, adr_net):
        super(ADRWrapper, self).__init__(env)
        self.env = env.unwrapped
        self.adr_net = adr_net
        self.original_mass = self.env.model.body_mass.copy()
        self.optimizer = optim.Adam(self.adr_net.parameters(), lr=0.001)
    
    def reset(self, **kwargs):
        state = self.env.sim.get_state()
        qpos = state.qpos
        qvel = state.qvel
        
        # Use ADR network to decide the randomization scale
        obs = self.env._get_obs()
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            random_scale = self.adr_net(obs_tensor).numpy()
        
        # Clip the random_scale to ensure values are in a valid range
        random_scale = np.clip(random_scale, 0.1, 10.0)
        
        self.env.model.body_mass[:] = self.original_mass * random_scale
        self.env.set_state(qpos, qvel)
        return self.env.reset(**kwargs)

    def update_adr(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def main():
    # Directories and filenames
    model_dir = './models'
    prefix = 'model_PPO'
    total_timesteps = 1000000
    adr_prefix = 'model_PPO_ADR'
    n_ep = 1000  # Number of additional episodes to train
    results_file = os.path.join(model_dir, 'results_training_ppo.txt')

    os.makedirs(model_dir, exist_ok=True)

    # Load the source environment
    source_env = gym.make('CustomHopper-source-v0')
    # Initialize the ADR network
    adr_net = ADRNetwork(input_dim=source_env.observation_space.shape[0], output_dim=1)
    # Initialize the ADR-wrapped target environment
    source_adr_env = ADRWrapper(gym.make('CustomHopper-source-v0'), adr_net)

    print('State space:', source_env.observation_space)
    print('Action space:', source_env.action_space)
    print('Dynamics parameters:', source_env.get_parameters())

    # Load the pre-trained PPO model
    model_path = os.path.join(model_dir, f'{prefix}_n_ts:{total_timesteps}.mdl')
    model = PPO.load(model_path, env=source_adr_env)
    print(f'Model loaded from {model_path}')

    # Continue training with ADR
    episode_rewards = []
    for ep in range(1,n_ep+1):
        if ep % 10 == 0:
            print(f"Episode number {ep}")
        
        obs = source_adr_env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = source_adr_env.step(action)
            total_reward += reward
        
        episode_rewards.append((total_timesteps + ep * source_adr_env.spec.max_episode_steps, total_reward))
        
        # Calculate ADR loss based on the reward (placeholder for actual loss calculation)
        adr_loss = torch.tensor(total_reward, dtype=torch.float32, requires_grad=True)

        # Update ADR network
        source_adr_env.update_adr(adr_loss)

        # Write the rewards to the file
        with open(results_file, 'a') as f:
            f.write(f"{total_timesteps + ep * source_adr_env.spec.max_episode_steps}\t{total_reward}\n")

        # Train the model
        model.learn(total_timesteps=source_adr_env.spec.max_episode_steps, reset_num_timesteps=False)

    # Save the final model
    model.save(os.path.join(model_dir, f'{adr_prefix}_n_ts:{total_timesteps}_n_ep:{n_ep}.mdl'))

    # Load the training rewards from the file
    timesteps = []
    rewards = []
    with open(results_file, 'r') as f:
        for line in f:
            timestep, reward = map(float, line.split('\t'))
            timesteps.append(timestep)
            rewards.append(reward)

    # Plot the training rewards
    plt.figure()
    plt.plot(timesteps[:len(timesteps)//2], rewards[:len(rewards)//2], 'b', label='PPO')
    plt.plot(timesteps[len(timesteps)//2:], rewards[len(rewards)//2:], 'r', label='PPO+ADR')
    plt.xlabel('Timesteps', fontsize=13)
    plt.ylabel('Mean Reward', fontsize=13)
    plt.title('Training Progress', fontsize=13)
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    plt.show()

if __name__ == '__main__':
    main()
