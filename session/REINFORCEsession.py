"""
A session is a Wrapper class for all the tasks of the project.
Create a session and call its methods to create, train and test an agent.
"""

from typing import Literal
from time import time

from agent.train_and_test.train import train

import torch

from env.custom_hopper import *
from agent.reinforce_agent import ReinforceAgent, Policy
from .session import Session

class REINFORCESession(Session):

    def __init__(
        self,
        env_path: Literal['CustomHopper-source-v0','CustomHopper-target-v0'],
		output_folder: str,
        verbose: int = 0,
        device: Literal['cpu', 'cuda'] = 'cpu'
    ):  
        super().__init__(env_path, output_folder, verbose, device)
    
    def get_step(self):
        return self._step
        
    def load_agent(
        self,
        policy_path: str = None,
        lr_policy: float = 1e-3,
        baseline: int = 0
    ):
        """Load a new reinforce with baseline agent in the session."""
        # Load the policy
        policy = Policy(self.state_space, self.action_space)
        if policy_path:
            policy.load_state_dict(torch.load(policy_path))

        self.agent = ReinforceAgent(policy, self.device, lr_policy, baseline)

        if self._verbose:
            print("Successful loading of the reinforce with baseline agent.")
        
        self.infos = {'lr_policy' : lr_policy, 'baseline' : baseline}

    def load_last_agent(self, lr_policy: float = 1e-3, baseline: int = 0, best:bool = True):
        best = "best_" if best else ""
        policy_path = f"self.output_folder/step_{self._step-1}_train/{best}model.mdl"
        try:
            self.load_agent(policy_path, lr_policy, baseline)
        except FileNotFoundError:
            self.load_agent(None, lr_policy, baseline)
            if self._verbose:
                print("Enable to fing the agent, created a new one instead.")

    
    def train(self, n_episode: int = 10000, early_stopping_threshold: int = None, episode0: int = 0):
        """
        Train the agent.

        Parameters:
        ----
        n_espisode: The number of maximum episodes of the training
        early_stopping_threshold: If the agent don't improve the cumulated reward during this number of episodes, stop the training.
        episode0: To be use only with verbose and fractionned training to not restart the episode count on printing.

        Return
        ----
        number_of_episodes: The number of episode of the training session (is n_episode with no early stopping)
        best_reward: The best cumulated reward achieve by the agent during the training
        """

        self._make_step_dir('train')

        starting_time = time()

        # Train the model
        episode_lengths, rewards, max_train_reward, best_model_episode = train(
            self.env,
            self.agent,
            n_episode,
            self._verbose,
            early_stopping_threshold,
            f"{self.output_folder}/step_{self._step}_train/",
            episode0
        )
        # Save the metrics
        self._save_metrics(rewards, episode_lengths, suffix='train', episode_of_best=best_model_episode)

        print(f"End of session step {self._step}, Lasted {(time() - starting_time):.2f} s, Best reward: {max_train_reward:.2f}")
        self._step += 1
        return episode_lengths, max_train_reward

    def train_agent_with_defined_moving_baseline(self, n_episodes_per_baseline: int, baselines: list[float]):
        """
        Train the agent with a baseline evolving.
        """
        for (i,baseline) in enumerate(baselines):
            self.agent.baseline = baseline
            self.train(n_episodes_per_baseline, episode0=i*n_episodes_per_baseline)
    
    def train_agent_with_increasing_goal_baseline(self, n_episodes_per_step, n_step, first_baseline):
        """
        Train an agent with mutliple steps by defining the baseline as the maximum reward of the previous step
        """
        self.agent.baseline = first_baseline
        for i in range(n_step):
            _, best_reward =self.train(n_episodes_per_step, episode0=i*n_episodes_per_step)
            self.agent.baseline = best_reward