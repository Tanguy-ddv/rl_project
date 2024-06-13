from typing import Literal
from .PPOsession import PPOSession
from adr.callback import ADRCallback
import os
from stable_baselines3.common.evaluation import evaluate_policy

class PPOADRSession(PPOSession):

    def __init__(self, output_folder: str, nenvs:int = 10, verbose: int = 0, device: Literal['cpu','cuda'] = 'cpu'):
        env_path = "ADRHopper-v0"
        self.ref_env_path = "CustomHopper-target-v0"
        self.nenvs = nenvs

        super().__init__(env_path, output_folder, verbose, device)
    
    def train(self, total_timesteps):
        """
        Train the PPO agent for a fixed number of timesteps
        BE CAREFUL, other agent are trained with a number of EPISODE,
        to retrieve the number of time steps, use the episode lengths.
        """
        os.mkdir(f"{self.output_folder}/step_{self._step}_train")
        callback = ADRCallback(self.ref_env_path, self.agent, nenvs=self.nenvs, output_folder=f"{self.output_folder}/step_{self._step}_train", verbose=self._verbose)
        self.agent.learn(total_timesteps=total_timesteps, callback=callback)
        self._step += 1