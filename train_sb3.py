"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
import numpy
numpy.save
from env.custom_randomized_hopper import *

from stable_baselines3 import PPO
register_uniform(1,4, "testuniform-v0")

def main():
    train_env = gym.make("CustomHopper-uniform-v0")

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    best_params = { # The best params found with the grid search for the source.
        'learning_rate': 0.00078,
        'n_steps': 2048,
        'batch_size': 64,
        'gamma': 0.99,
        'clip_range': 0.25
    }
    model = PPO("MlpPolicy", train_env, verbose=0)
    model.learn(total_timesteps=1)

    vec_env = model.get_env()

    obs = vec_env.reset()

    cumul_reward = 0
    done = False
    i = 0

    while not done and i < 10_000:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()
        cumul_reward += reward
        i+=1

    print(cumul_reward)
    print(i)
    model.save("uniform_model.mdl")

if __name__ == '__main__':
    main()