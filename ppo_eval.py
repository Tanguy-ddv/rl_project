import gym
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

def train_ppo(train_env: gym.Env, name: str, total_timesteps, ppo_kwargs = {}, seed = 42):

    model = PPO("MlpPolicy", train_env, verbose=0, **ppo_kwargs, seed=seed)
    model.learn(total_timesteps=total_timesteps)
    model.save(name)
    return model