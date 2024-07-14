from env import *

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class UDRCallback(BaseCallback):
    """Only to use on UDRHopper."""

    def __init__(self, model: PPO, output_folder: str = "", delta: float = 1, verbose=0):
        super().__init__(verbose)
        self.delta = delta


        self.init_callback(model)

    def _on_step(self) -> bool:

        done = self.locals['dones'][0] # bool
        if done:
            # We randomise the environement.
            self.training_env.envs[0].modify_parameters(self.delta)

        return True

class GDRCallback(BaseCallback):
    """Only to use on GDRHopper"""

    def __init__(self, model: PPO, output_folder: str = "", delta: float = 0.2, verbose: int = 0, ):
        super().__init__(verbose)
        self.delta = delta

        self.init_callback(model)

    def _on_step(self) -> bool:

        done = self.locals['dones'][0] # bool
        if done:
            # We randomise the environement.
            self.training_env.envs[0].modify_parameters(delta=self.delta)
        
        return True
    