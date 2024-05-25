import gym
from stable_baselines3 import PPO

def train_ppo(train_env: gym.Env, name: str, total_timesteps, ppo_kwargs = {}):

    model = PPO("MlpPolicy", train_env, verbose=0, **ppo_kwargs)
    model.learn(total_timesteps=total_timesteps)
    model.save(name)
    return model