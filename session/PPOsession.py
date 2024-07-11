from typing import Literal
from .session import Session
from stable_baselines3 import PPO
import gym
import os
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

class PPOSession(Session):

    def __init__(self, env_path: Literal['CustomHopper-source-v0','CustomHopper-target-v0'], output_folder: str, verbose: int = 0, device: Literal['cpu','cuda'] = 'cpu'):
        super().__init__(env_path, output_folder, verbose, device)
    
    def load_agent(self, path: str=None, **ppo_kwargs):
        """
        Load a PPO agent
        """
        if path is None:
            self.agent= PPO('MlpPolicy', self.env, **ppo_kwargs)
        else:
            self.agent = PPO.load(path, self.env, device = self.device)
        
        self.infos = ppo_kwargs
        self.callbacks = []
        
        
    def load_last_agent(self):
        actor_path = f"{self.output_folder}/step_{self._step-1}_train/model.mdl"
        try:
            self.load_agent(actor_path)
        except FileNotFoundError:
            raise ValueError("Enable to fing the agent, please verify the path and try again.")
    
    def load_callback(self, callback_class, **kwargs):
        """
        Add a new callback for the training. At the end of the training, clear the callback list.
        """
        output_folder = f"{self.output_folder}/step_{self._step}_train"
        self.callbacks.append(callback_class(model=self.agent, output_folder=output_folder, verbose=self._verbose, **kwargs))

    def train(self, total_timesteps):
        """
        Train the PPO agent for a fixed number of timesteps
        BE CAREFUL, other agent are trained with a number of EPISODE,
        to retrieve the number of time steps, use the episode lengths.
        """
        os.mkdir(f"{self.output_folder}/step_{self._step}_train")
        self.agent.learn(total_timesteps=total_timesteps, callback=CallbackList(self.callbacks))
        self.agent.save(f"{self.output_folder}/step_{self._step}_train/model.mdl")
        self._step += 1
        self.callbacks.clear()
    
    def test(self, n_episodes: int, env_path:str = None):
        """
        Test the agent on a given env.
        if env_path is None, test on the training env.
        """
        if env_path is not None:
            env = gym.make(env_path)
        else:
            env = self.env
        mean_reward, std_reward = evaluate_policy(self.agent, env, n_episodes, deterministic=True)
        self._save_metrics(rewards= mean_reward, episode_lengths= None, suffix='test')
        print(f'End of test (step={self._step}), Mean return = {mean_reward:.2f}, Std return = {std_reward:.4f}')
        self._step += 1


    
