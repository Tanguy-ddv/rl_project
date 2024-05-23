"""
A session is a Wrapper class for all the tasks of the project.
Create a session and call its methods to create, train and test an agent.
"""

from typing import Literal
from time import time
import shutil

from train import train
from test_agent import test

import torch
import gym
import json
import os

from env.custom_hopper import *
from agent.reinforce_agent import ReinforceAgent, Policy
from agent.actor_critic_agent import ActorCriticAgent, Critic

# agent types
_ACTOR_CRITIC = 1
_REINFORCE_WITH_BASELINE = 2

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

class Session:

    def __init__(
        self,
        env_path: Literal['CustomHopper-source-v0','CustomHopper-target-v0'],
		output_folder: str,
        verbose: int = 0,
        device: Literal['cpu', 'cuda'] = 'cpu'
    ):  
        self.__verbose = verbose
        self.output_folder = output_folder
        self.device = device

        if not os.path.isdir(output_folder):
            self.__step = 0
            os.mkdir(output_folder)
        else:
            self.__step = _find_last_step(output_folder)+1

        # Creation of the environment
        self.env = gym.make(env_path)

        self.state_space = self.env.observation_space.shape[-1]
        self.action_space = self.env.action_space.shape[-1]
        self.agent_type = None
    
        if verbose:

            print(f"Successful creation of the session, first step is step={self.__step}.")
            
            if verbose >= 2:
                print('Action space:', self.env.action_space)
                print('State space:', self.env.observation_space)
                print('Dynamics parameters:', self.env.get_parameters())
    
    def get_step(self):
        return self.__step
    
    def load_actor_critic(
            self,
            actor_path: str = None,
            critic_path: str = None,
            lr_actor: float=1e-3,
            lr_critic: float=1e-3
        ):
        """Load a new actor-critic agent in the session."""
        # Load the actor
        actor = Policy(self.state_space, self.action_space)
        if actor_path:
            actor.load_state_dict(torch.load(actor_path))

        # Load the critic
        critic = Critic(self.state_space)
        if critic_path:
            critic.load_state_dict(torch.load(critic_path))

        self.agent = ActorCriticAgent(actor, critic, self.device, lr_actor, lr_critic)
        self.agent_type = _ACTOR_CRITIC

        if self.__verbose:
            print("Successful loading of the actor-critic agent.")
        
        self.infos = {'type' : _ACTOR_CRITIC, 'lr_actor' : lr_actor, 'lr_critic' : lr_critic}
    
    def load_last_actor_critic(self, lr_actor: float=1e-3, lr_critic:float=1e-3, best:bool = True):
        best = "best_" if best else ""
        actor_path = f"self.output_folder/step_{self.__step-1}_train/{best}model.mdl"
        critic_path =  f"self.output_folder/step_{self.__step-1}_train/{best}critic.mdl"
        try:
            self.load_actor_critic(actor_path, critic_path, lr_actor, lr_critic)
        except FileNotFoundError:
            self.load_actor_critic(None, None, lr_actor, lr_critic)
        
    def load_reinforce_with_baseline(
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
        self.agent_type = _REINFORCE_WITH_BASELINE

        if self.__verbose:
            print("Successful loading of the reinforce with baseline agent.")
        
        self.infos = {'type' : _ACTOR_CRITIC, 'lr_policy' : lr_policy, 'baseline' : baseline}

    def load_last_reinforce(self, lr_policy: float = 1e-3, baseline: int = 0, best:bool = True):
        best = "best_" if best else ""
        policy_path = f"self.output_folder/step_{self.__step-1}_train/{best}model.mdl"
        try:
            self.load_reinforce_with_baseline(policy_path, lr_policy, baseline)
        except FileNotFoundError:
            self.load_reinforce_with_baseline(None, lr_policy, baseline)


    def __save_metrics(self, rewards, final_rewards, states, episode_lengths, suffix: str):

        os.mkdir(f"{self.output_folder}/step_{self.__step}_{suffix}")

        with open(f"{self.output_folder}/step_{self.__step}_{suffix}/rewards.json",'w') as f:
            json.dump(rewards, f)

        with open(f"{self.output_folder}/step_{self.__step}_{suffix}/final_rewards.json",'w') as f:
            json.dump(final_rewards, f)

        with open(f"{self.output_folder}/step_{self.__step}_{suffix}/episode_lengths.json",'w') as f:
            json.dump(episode_lengths, f)

        with open(f"{self.output_folder}/step_{self.__step}_{suffix}/states.json",'w') as f:
            json.dump(states, f)
    
    def store_infos(self, infos):
        with open(self.output_folder + '/infos.txt', 'a') as f:
            f.write(str(infos) + '\n')
    
    def clear_outputs(self):
        """
        Clear the output folder of previous trainings.
        """
        self.__step = 0
        shutil.rmtree(self.output_folder)
        os.mkdir(self.output_folder)

    def test_agent(self, n_episode: int = 10, render: bool = True):
        """
        Test the agent on the environment
        """
        if not self.agent_type:
            raise RuntimeError("No agent loaded yet, please load a agent first")

        starting_time = time()

        rewards, episode_lengths, final_rewards, states, max_train_reward = test(
            self.env, self.agent, n_episode, render
        )
        
        print(f"Average reward: {(sum(final_rewards)/n_episode):.2f} | Average episode Length: {(sum(episode_lengths)/n_episode):.2f} | Maximum train reward: {max_train_reward:.2f}")
        self.__save_metrics(rewards, final_rewards, states, episode_lengths, suffix='test')
        print(f"End of session step {self.__step}, lasted {(time() - starting_time):.2f} s")
        self.__step += 1
    
    def train_agent(self, n_episode: int = 10000, early_stopping_threshold: int = None):
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
        if not self.agent_type:
            raise RuntimeError("No agent loaded yet, please load a agent first")

        starting_time = time()

        # Train the model
        rewards, episode_lengths, final_rewards, states, max_train_reward = train(
            self.env, self.agent, n_episode, self.__verbose, early_stopping_threshold
        )
        # Save the metrics
        self.__save_metrics(rewards, final_rewards, states, episode_lengths, suffix='train')

        # Move the best model to the good folder
        shutil.move("best_model.mdl", f"{self.output_folder}/step_{self.__step}_train/best_model.mdl")
        shutil.move("episode_of_best.txt", f"{self.output_folder}/step_{self.__step}_train/episode_of_best.txt")

        if os.path.exists("best_critic.mdl"):
            shutil.move("best_critic.mdl", f"{self.output_folder}/step_{self.__step}_train/best_critic.mdl")

        # Save the current model.
        torch.save(self.agent.policy.state_dict(), f"{self.output_folder}/step_{self.__step}_train/model.mdl")
        if self.agent_type == _ACTOR_CRITIC:
            torch.save(self.agent.critic.state_dict(), f"{self.output_folder}/step_{self.__step}_train/critic.mdl")

        print(f"End of session step {self.__step}, Lasted {(time() - starting_time):.2f} s, Best reward: {max_train_reward:.2f}")
        self.__step += 1
        return len(episode_lengths), max_train_reward

    def train_agent_with_early_stopping(self,  n_episodes: int = 1000, early_stopping_threshold: int = 50, max_nb_restarts: int = 100):
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
                if self.agent_type == _ACTOR_CRITIC:
                    self.load_last_actor_critic(
                        lr_critic=self.infos['lr_critic'],
                        lr_actor=self.infos['lr_actor'],
                        best=True
                    )
                else:
                    self.load_last_reinforce(
                        lr_policy=self.infos['lr_policy'],
                        baseline=self.infos['baseline'],
                        best=True
                    )
            # Train it again until the threshold is reached
            this_step_n_episodes, _ = self.train_agent(n_episodes - total_episodes, early_stopping_threshold)
            total_episodes += this_step_n_episodes - early_stopping_threshold
            nb_restarts+=1
            print(f"Still {n_episodes - total_episodes - early_stopping_threshold} to go")

    def train_agent_with_defined_moving_baseline(self, n_episodes_per_baseline: int, baselines: list[int]):
        """
        Train the agent with a baseline evolving.
        """
        if self.agent_type != _REINFORCE_WITH_BASELINE:
            raise RuntimeError("This training method is only provided for reinforce agent.")
        for baseline in baselines:
            self.agent.baseline = baseline
            self.train_agent(n_episodes_per_baseline)
    
    def train_agent_with_increasing_goal_baseline(self, n_episodes_per_step, n_step, first_baseline):
        """
        Train an agent with mutliple steps by defining the baseline as the maximum reward of the previous step
        """
        if self.agent_type != _REINFORCE_WITH_BASELINE:
            raise RuntimeError("This training method is only provided for reinforce agent.")
        self.agent.baseline = first_baseline
        for _ in range(n_step):
            _, best_reward =self.train_agent(n_episodes_per_step)
            self.agent.baseline = best_reward