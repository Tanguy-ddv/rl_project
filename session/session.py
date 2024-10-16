"""
A session is a Wrapper class for all the tasks of the project.
Create a session and call its methods to create, train and test an agent.
"""
import json
import os
from typing import Literal
from abc import ABC, abstractmethod
from time import time
import env # Register all the envs here
from agent.train_and_test import test
import shutil
import matplotlib.pyplot as plt
from rolling_avg import rolling_avg

import gym

def _find_last_step(path_folder):
    """
    During a session, several steps are executed. This function find the last step number.
    """
    max_step = 0
    for element in os.listdir(path_folder):
        if os.path.isdir(os.path.join(path_folder, element)) and element.startswith('step_'):
            step = int(element.split('_')[1])
            if step > max_step:
                max_step = step
    return max_step

class Session(ABC):

    def __init__(
        self,
        env_path: Literal['CustomHopper-source-v0','CustomHopper-target-v0'],
		output_folder: str,
        verbose: int = 0,
        device: Literal['cpu', 'cuda'] = 'cpu'
    ):  
        self._verbose = verbose
        self.output_folder = output_folder
        self.device = device

        if not os.path.isdir(output_folder):
            self._step = 0
            os.mkdir(output_folder)
        else:
            self._step = _find_last_step(output_folder)+1

        # Creation of the environment
        self.env: env.CustomHopper = gym.make(env_path)

        self.state_space = self.env.observation_space.shape[-1]
        self.action_space = self.env.action_space.shape[-1]
    
        if verbose:

            print(f"Successful creation of the session, first step is step={self._step}.")
            
            if verbose >= 2:
                print('Action space:', self.env.action_space)
                print('State space:', self.env.observation_space)
                print('Dynamics parameters:', self.env.get_parameters())
            
        self.loaded_agent = False

        self.agent = None
    
    def get_step(self):
        return self._step
    
    @abstractmethod
    def load_agent(self, path:str):
        """
        Load an agent saved at the given path.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def load_last_agent(self, best: bool = True):
        """
        Load the last agent stored by the session.
        If best is true, load the last best agent, else, load the last fully trained agent.
        """
        raise NotImplementedError()

    def _make_step_dir(self, suffix:str):
        try:
            os.mkdir(f"{self.output_folder}/step_{self._step}_{suffix}")
        except FileExistsError:
            pass

    def _save_metrics(self, rewards: list[float], episode_lengths: list[float], suffix: str, episode_of_best: int = None, other_dict: dict = {}):

        self._make_step_dir(suffix)

        if rewards is not None:
            with open(f"{self.output_folder}/step_{self._step}_{suffix}/rewards.json",'w') as f:
                json.dump(rewards, f)

        if episode_lengths is not None:
            with open(f"{self.output_folder}/step_{self._step}_{suffix}/episode_lengths.json",'w') as f:
                json.dump(episode_lengths, f)
        
        if episode_of_best is not None:
            with open(f"{self.output_folder}/step_{self._step}_{suffix}/episode_of_best.json",'w') as f:
                json.dump(episode_of_best, f)
        
        for metric in other_dict:
            with open(f"{self.output_folder}/step_{self._step}_{suffix}/{metric}.json",'w') as f:
                json.dump(other_dict[metric], f)
    
    def store_infos(self, infos):
        with open(self.output_folder + '/infos.txt', 'a') as f:
            f.write(str(infos) + '\n')
    
    def clear_outputs(self):
        """
        Clear the output folder of previous trainings.
        """
        self._step = 0
        shutil.rmtree(self.output_folder)
        os.mkdir(self.output_folder)

    def test(self, n_episode: int = 10, render: bool = True, seed:int = None):
        """
        Test the agent on the environment
        """

        self._make_step_dir('test')

        starting_time = time()

        rewards, episode_lengths, max_train_reward = test(
            self.env, self.agent, n_episode, render, seed
        )
        
        print(f"Average reward: {(sum(rewards)/n_episode):.2f} | Average episode Length: {(sum(episode_lengths)/n_episode):.2f} | Maximum train reward: {max_train_reward:.2f}")
        self._save_metrics(rewards, episode_lengths, suffix='test')
        print(f"End of session step {self._step}, lasted {(time() - starting_time):.2f} s")
        self._step += 1
    
    @abstractmethod
    def train(self):
        """
        Train the agent.
        """
        raise NotImplementedError()

    def plot_metric(self, metric: str, steps: list, n_avg = 100, suffix='train', y_scale_log: bool = False):
        """Plot a metric cumulated over several episodes."""

        fig, ax = plt.subplots(1,1)
        for step in steps:
            dir = self.output_folder + f'/step_{step}_{suffix}/'
            with open(dir + f'{metric}.json') as f:
                rewards = json.load(f)
            averaged_rewards = rolling_avg(rewards, n_avg)
            ax.plot(range(0, len(averaged_rewards)*n_avg, n_avg), averaged_rewards)
        ax.set_xlabel("Episode", fontsize=13)
        ax.set_ylabel(f"Mean of {metric} over {n_avg} episodes", fontsize=13)
        ax.set_xlabel("Episode", fontsize=13)
        ax.tick_params(axis='both', labelsize=12)
        if len(steps) > 1:
            ax.legend(fontsize=12)
        if y_scale_log:
            ax.set_yscale('log')
        return fig, ax