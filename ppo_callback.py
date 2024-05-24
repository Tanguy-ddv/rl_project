"""code copy/paste (and slightly modified) from the stable-baselines3 documentation."""

import os

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback


class StoreReward(BaseCallback):
    """
    Callback for saving a model (the check is done every `check_freq` steps)
    based on the training reward (in practice, we recommend using `EvalCallback`).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the `Monitor` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def _init_(self, check_freq: int, log_dir: str, verbose: int = 1):
        super()._init_(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.rewards = []

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Retrieve training reward
        df = load_results(self.log_dir)
        print(df)
        if len(x) > 0:
            # Mean training reward over the last 100 episodes
            self.rewards.append(np.mean(y[-1]))

        return True

    def _on_training_end(self) -> None:
        print(self.rewards)