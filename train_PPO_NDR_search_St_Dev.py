import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from env import *
from domain_randomization.callbacks import GDRCallback

seed_value = 42

def main():

    std_devs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.5, 0.6, 0.7, 0.8]
    std_devs_str = ','.join(map(str, std_devs))

    total_timesteps = 500000
    n_episodes = 1000

    models_dir = f"models_NDR_seed:{seed_value}"
    os.makedirs(models_dir, exist_ok=True)

    results_file_path = os.path.join(models_dir, f"mean_rewards_seed:{seed_value}.txt")

    for std_dev in std_devs:

        random_source_env = gym.make(NDR)
        random_source_env.seed(seed_value)

        model_dir = f"ppo_source_NDR_ep:{total_timesteps}_stdev:{std_dev}_nEpEval:{n_episodes}_seed:{seed_value}"
        model_path = os.path.join(models_dir, model_dir)

        if not os.path.exists(model_path):
            model_standard = PPO("MlpPolicy", env=random_source_env, device='cpu', verbose=0, seed=seed_value)
            callback = GDRCallback(model_standard, delta=std_dev)
            model_standard.learn(total_timesteps=total_timesteps, callback=callback)
            model_standard.save(model_path)

            model = PPO.load(model_path)

            test_env_target = gym.make('CustomHopper-target-v0')
            test_env_target.seed(seed_value)

            target_mean_reward, _ = evaluate_policy(model, test_env_target, n_eval_episodes=n_episodes, render=True)

            with open(results_file_path, "a") as results_file:
                results_file.write(f"{std_dev} {target_mean_reward:.2f}\n")

            print(f"Domain Randomization with std_dev = {std_dev} :")
            print(f"name of the model directory: {model_dir}")
            print(f'Mean reward of standard model on NRD in target test env : {target_mean_reward}\n')

    std_devs_from_file = []
    results_target_from_file = []

    with open(results_file_path, "r") as results_file:
        for line in results_file:
            std_dev, mean_reward = map(float, line.strip().split())
            std_devs_from_file.append(std_dev)
            results_target_from_file.append(mean_reward)

    # Sort data for plotting
    std_devs_from_file, results_target_from_file = zip(*sorted(zip(std_devs_from_file, results_target_from_file)))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(std_devs_from_file, results_target_from_file, marker='o', markersize=10, linewidth=2, label='Mean Reward', color='tab:blue')
    plt.xlabel('Standard Deviation', fontsize=15)
    plt.ylabel('Test Mean Reward', fontsize=15)
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Save plot
    plot_path = os.path.join(models_dir, f"plot_with_std_devs_{std_devs_str}.png")
    plt.savefig(plot_path)
    plt.show()

if __name__ == '__main__':
    main()
