"""
A session is a Wrapper class for all the tasks of the project.
Create a session and call its methods to create, train and test an agent.
"""

from typing import Literal
from time import time

from agent.train_and_test import train, test

import torch
import gym

from agent.actor_critic_agent import ActorCriticAgent, Critic, Actor
from .session import Session

class ACSession(Session):

    def __init__(
        self,
        env_path: Literal['CustomHopper-source-v0','CustomHopper-target-v0'],
		output_folder: str,
        verbose: int = 0,
        device: Literal['cpu', 'cuda'] = 'cpu'
    ):  
        super().__init__(env_path, output_folder, verbose, device)

    def get_step(self):
        """Return the step the session is in."""
        return self._step
    
    def load_agent(
            self,
            actor_path: str = None,
            critic_path: str = None,
            lr_actor: float=1e-3,
            lr_critic: float=1e-3
        ):
        """Load a new actor-critic agent in the session."""
        # Load the actor
        actor = Actor(self.state_space, self.action_space)
        if actor_path:
            actor.load_state_dict(torch.load(actor_path))

        # Load the critic
        critic = Critic(self.state_space)
        if critic_path:
            critic.load_state_dict(torch.load(critic_path))

        self.agent = ActorCriticAgent(actor, critic, self.device, lr_actor, lr_critic)

        if self._verbose:
            print("Successful loading of the actor-critic agent.")
        
        self.infos = {'lr_actor' : lr_actor, 'lr_critic' : lr_critic}
    
    def load_last_agent(self, lr_actor: float=1e-3, lr_critic:float=1e-3, best:bool = True):
        best = "best_" if best else ""
        actor_path = f"{self.output_folder}/step_{self._step-1}_train/{best}model.mdl"
        critic_path =  f"{self.output_folder}/step_{self._step-1}_train/{best}critic.mdl"
        try:
            self.load_agent(actor_path, critic_path, lr_actor, lr_critic)
        except FileNotFoundError:
            self.load_agent(None, None, lr_actor, lr_critic)
            if self._verbose:
                print("Enable to fing the agent, created a new one instead.")
    
    def train(self, n_episode: int = 10000, early_stopping_threshold: int = None):
        """
        Train the agent.

        Parameters:
        ----
        n_espisode: The number of maximum episodes of the training
        early_stopping_threshold: If the agent don't improve the cumulated reward during this number of episodes, stop the training.
        
        Return
        ----
        number_of_episodes: The number of episode of the training session (is n_episode with no early stopping)
        best_reward: The best cumulated reward achieve by the agent during the training
        """

        self._make_step_dir('train')

        starting_time = time()

        # Train the model
        rewards, episode_lengths, final_rewards, states, max_train_reward = train(
            self.env, self.agent, n_episode, self._verbose, early_stopping_threshold, f"{self.output_folder}/step_{self._step}_train/"
        )
        # Save the metrics
        self._save_metrics(rewards, final_rewards, states, episode_lengths, suffix='train')

        print(f"End of session step {self._step}, Lasted {(time() - starting_time):.2f} s, Best reward: {max_train_reward:.2f}")
        self._step += 1
        return len(episode_lengths), max_train_reward

    def train_with_early_stopping(self,  n_episodes: int = 1000, early_stopping_threshold: int = 50, max_nb_restarts: int = 100):
        """
        Train the agent with checkpointing: every time the agent don't improve during 'early_stopping_threshold'
        episodes, get back to the best policy and try again from this point.

        Parameters:
        n_espisode: The number of maximum episodes of the training
        early_stopping_threshold: If the agent don't improve the cumulated reward during this number of episodes, restart the training from the last best policy model.
        max_nb_restarts: The number maximum of 
        """
        total_episodes = 0
        nb_restarts = 0
        n_episodes += early_stopping_threshold
        while total_episodes<n_episodes - early_stopping_threshold and nb_restarts < max_nb_restarts:
            
            if total_episodes != 0:
                # Load the last best actor from the previous step.
                self.load_last_agent(
                    lr_critic=self.infos['lr_critic'],
                    lr_actor=self.infos['lr_actor'],
                    best=True
                )
            # Train it again until the threshold is reached
            this_step_n_episodes, _ = self.train(n_episodes - total_episodes, early_stopping_threshold)
            total_episodes += this_step_n_episodes - early_stopping_threshold
            nb_restarts+=1
            print(f"Still {n_episodes - total_episodes - early_stopping_threshold} to go")
    
    def train_and_test_on_other_env(
            self,
            other_env_path: str,
            n_train_episodes: int = 1000,
            test_every: int = 100,
            nb_test: int = 10,
            ):
        """
        Train the agent and test it on another env during the training to see the evolution of the performance on both envs.
        """

        test_env = gym.make(other_env_path)
        self._make_step_dir('train')

        