import os
import gym
import matplotlib.pyplot as plt
from env.custom_randomized_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

class RewardLoggerCallback(BaseCallback):
    def __init__(self, check_freq, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.rewards = []
        self.episode_rewards = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Log mean reward at regular intervals
        if self.n_calls % self.check_freq == 0:
            mean_reward, std_reward = evaluate_policy(self.model, self.training_env, n_eval_episodes=10, render=False)
            self.rewards.append((self.n_calls, mean_reward))
            if self.verbose > 0:
                print(f"Num timesteps: {self.n_calls} - Mean reward: {mean_reward:.2f} - Std reward: {std_reward:.2f}")
        
        # Log reward per episode
        if len(self.locals['infos']) > 0 and 'episode' in self.locals['infos'][0]:
            episode_reward = self.locals['infos'][0]['episode']['r']
            self.episode_rewards.append(episode_reward)
            self.episode_count += 1
            print(f"Episode {self.episode_count} - Reward: {episode_reward}")
        
        return True

def main():
    # Directories and filenames
    model_dir = './models'
    prefix = 'model_PPO'
    results_file = os.path.join(model_dir, 'results_training_ppo.txt')

    # Set environment
    source_env = gym.make('CustomHopper-source-v0')
    target_env = gym.make('CustomHopper-target-v0')
    print('State space:', source_env.observation_space)
    print('Action space:', source_env.action_space)
    print('Dynamics parameters:', source_env.get_parameters())

    # Define total timesteps for training
    total_timesteps = 1000000

    # Create the PPO model
    model = PPO("MlpPolicy", source_env, verbose=0)

    # Create the reward logger callback
    check_freq = 10000
    callback = RewardLoggerCallback(check_freq, verbose=1)

    # Train the model
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, callback=callback)

    # Save the model with the total timesteps used
    model.save(os.path.join(model_dir, f'{prefix}_n_ts:{total_timesteps}.mdl'))
    print(f'Model saved with {total_timesteps} timesteps.')

    # Save the training rewards to a text file
    with open(results_file, 'w') as f:
        for timestep, reward in callback.rewards:
            f.write(f"{timestep}\t{reward}\n")
    print(f'Training rewards saved to {results_file}.')

    # Plot the training rewards
    timesteps, rewards = zip(*callback.rewards)
    plt.figure()
    plt.plot(timesteps, rewards, label='Mean Reward')
    plt.xlabel('Timesteps', fontsize=13)
    plt.ylabel('Mean Reward', fontsize=13)
    plt.title('Training Progress', fontsize=13)
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    plt.show()

    num_episodes = 1000

    # Evaluate the model on source environment
    print("Evaluation on Source:")
    mean_reward, std_reward = evaluate_policy(model, source_env, n_eval_episodes=num_episodes, render=False)
    print(f'Evaluation over {num_episodes} episodes on source environment:')
    print(f'Mean Reward: {mean_reward}, Std Reward: {std_reward}')

    # Evaluate the model on target environment
    print("Evaluation on Target:")
    mean_reward, std_reward = evaluate_policy(model, target_env, n_eval_episodes=num_episodes, render=False)
    print(f'Evaluation over {num_episodes} episodes on target environment:')
    print(f'Mean Reward: {mean_reward}, Std Reward: {std_reward}')

if __name__ == '__main__':
    main()

