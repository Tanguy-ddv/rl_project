import numpy as np
import gym
from sb3_contrib import RecurrentPPO
from env.custom_randomized_hopper import *
from stable_baselines3.common.evaluation import evaluate_policy
env = gym.make("CustomHopper-source-v0")
best_params = { # The best params found with the grid search for the source.
    'learning_rate': 0.00078,
    'n_steps': 2048,
    'batch_size': 64,
    'gamma': 0.99,
    'clip_range': 0.25
}
model = RecurrentPPO("MlpLstmPolicy", env, verbose=2, **best_params)
model.learn(500000)

test_env = gym.make("CustomHopper-target-v0")
mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=20, warn=False, render=False)
print(mean_reward)

model.save("ppo_recurrent")