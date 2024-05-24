"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
import numpy
numpy.save
from typing import Literal
from env.custom_randomized_hopper import *
from ppo_callback import StoreReward

from stable_baselines3 import PPO, SAC, A2C
register_uniform(0,4, "testuniform-v0")

def make_env(domain: Literal['uniform', 'normal', 'target', 'soucre', None] = None):
    return gym.make(f'CustomHopper-{domain}-v0')

def main():
    train_env = gym.make("testuniform-v0")

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    best_params = {
        'learning_rate': 0.00078,
        'n_steps': 2048,
        'batch_size': 64,
        'gamma': 0.99,
        'clip_range': 0.25
    }
    model = PPO("MlpPolicy", train_env, verbose=1, **best_params)
    model.learn(total_timesteps=1000)

    vec_env = model.get_env()
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

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
    test_env = make_env('target')

    print('State space:', test_env.observation_space)  # state-space
    print('Action space:', test_env.action_space)  # action-space
    print('Dynamics parameters:', test_env.get_parameters())  # masses of each link of the Hopper

    best_params = {
        'learning_rate': 0.00078,
        'n_steps': 2048,
        'batch_size': 64,
        'gamma': 0.99,
        'clip_range': 0.25
    }
    model = PPO.load("uniform_model.mdl", test_env)

    obs = test_env.reset()

    cumul_reward = 0
    done = False
    i = 0

    while not done and i < 10_000:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        test_env.render()
        cumul_reward += reward
        i+=1

    print(cumul_reward)
    print(i)
    model.save("uniform_model.mdl")