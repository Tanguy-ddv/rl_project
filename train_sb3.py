"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *

from stable_baselines3 import PPO, SAC

def main():
    train_env = gym.make('CustomHopper-target-v0')

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
    model.learn(total_timesteps=500_000)

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
    model.save("best_ppo_target.mdl")
    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #

if __name__ == '__main__':
    #main()
    train_env: gym.Env = gym.make('CustomHopper-source-v0')
    model = PPO.load("best_ppo_target.mdl", train_env)
    obs = train_env.reset()
    
    cumul_reward = 0
    done = False
    i = 0
    
    while not done and i < 10_000:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = train_env.step(action)
        train_env.render()
        cumul_reward += reward
        i+=1
    
    print(cumul_reward)
    print(i)