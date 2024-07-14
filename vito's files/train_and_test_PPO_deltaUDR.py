import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from env.custom_randomized_hopper import register_uniform_with_delta

seed_value = 42
deltas = [0.25, 0.5, 0.75]
total_timesteps = 5000000
n_episodes = 10
eval_freq = 5000

models_dir = f"models_delta_UDR_5M_seed:{seed_value}"
os.makedirs(models_dir, exist_ok=True)

def main():
    results = {delta: [] for delta in deltas}
    timesteps = np.arange(eval_freq, total_timesteps + eval_freq, eval_freq)

    results_file_path = os.path.join(models_dir, "result.txt")
    with open(results_file_path, "a") as results_file:
        for delta in deltas:
            name = f"CustomHopper-uniform-with_delta-{delta}-v0"
            register_uniform_with_delta(delta, name=name)
            
            random_source_env = gym.make(name)
            random_source_env.seed(seed_value)
            
            model = PPO("MlpPolicy", env=random_source_env, device='cpu', verbose=0, seed=seed_value)
            
            test_env_target = gym.make('CustomHopper-target-v0')
            test_env_target.seed(seed_value)

            for timestep in timesteps:
                model.learn(total_timesteps=eval_freq, reset_num_timesteps=False)

                mean_reward, _ = evaluate_policy(model, test_env_target, n_eval_episodes=n_episodes, render=False)
                results[delta].append(mean_reward)
                log_message = f"Delta: {delta}, Timesteps: {timestep}, Mean Reward: {mean_reward}\n"
                print(log_message)
                results_file.write(log_message)

    # Plot results
    plt.figure(figsize=(12, 6))
    for delta in deltas:
        plt.plot(timesteps, results[delta], marker='o', markersize=5, linewidth=2, label=f'Delta: {delta}')

    plt.xlabel('Training Timesteps')
    plt.ylabel('Mean Reward of Test in Target Environment')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(models_dir, "performance_plot.png")
    plt.savefig(plot_path)
    plt.show()

if __name__ == '__main__':
    main()
