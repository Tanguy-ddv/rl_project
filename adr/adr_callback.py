from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.ppo import PPO
from .particle import Particle
from .discriminator import Discriminator
import torch 
import json
import gym
import os
from agent.train_and_test import performPPO

class ADRCallback(BaseCallback):

    def __init__(
        self,
        ref_env_path: str,
        model: PPO,
        nenvs: int,
        output_folder: str,
        lr_discriminator: float=1e-3,
        lr_particle: float=1e-3,
        verbose: int = 0
    ):
        super().__init__(verbose)

        self.ref_env = gym.make(ref_env_path)
        self.nenvs = nenvs
        self.output_folder = output_folder
        self.lr_discriminator = lr_discriminator
        self.lr_particle = lr_particle

        self.init_callback(model)

    def _init_callback(self) -> None:

        state_space = self.model.env.observation_space.shape[-1]
        action_space = self.model.env.action_space.shape[-1]
        self.discriminator = Discriminator(state_space, action_space, 64, self.lr_discriminator)
        
        self.__current_env = 0
        
        # No get_parameters on DummyVecEnv so have to get the values this way.
        self.mean_values = self.training_env.envs[0].get_parameters()
        self.nparams = self.mean_values.shape[0]

        self.particles = [Particle(self.nparams, self.mean_values, hidden=64, lr=self.lr_particle) for _ in range(self.nenvs)]
        
        self.ref_rewards = []
        self.rd_rewards = []
        self.episode_lengths = []

        self.current_episode_length = 0

        self.states = []
        self.previous_states = []
        self.actions = []
        self.parameters = self.particles[0].values
        self.current_reward = 0
    
    def __update_current_env_id(self):
        self.__current_env = (self.__current_env+1)%self.nenvs

    def _on_step(self):
        """Action to be done at each step."""

        # Retrieve the data
        previous_state = self.locals['obs_tensor'][0].numpy() # np.ndarray
        state = self.locals['new_obs'][0] # np.ndarray
        action = self.locals['actions'][0] # np.ndarray
        reward = self.locals['rewards'][0] # float
        log_probs = self.locals['log_probs'].detach() # float
        done = self.locals['dones'][0] # bool

        # Store them
        self.states.append(state)
        self.previous_states.append(previous_state)
        self.actions.append(action)
        self.current_reward += reward
        self.current_episode_length += 1

        if done:
            # Test the agent on the ref environement and update the discriminator.
            if self.__current_env == self.nenvs-1: 
                
                ref_reward = 0
                previous_states, actions, states, ref_reward, episode_length = performPPO(self.ref_env, self.model, False)

                for (previous_state, state, action) in zip(previous_states, states, actions): 
                    # The done and the reward and the log_probs are not used on the training of the discriminator so we put whatever we want
                    self.discriminator.store_ref_outcome(previous_state, state, 0, 0, None, action)

                self.ref_rewards.append(ref_reward)
                self.episode_lengths.append(episode_length)

                self.discriminator.update_policy()
                self.discriminator.clear_history()

            # Update current particle
            for state, previous_state, action in zip(self.states, self.previous_states, self.actions):
                self.discriminator.store_outcome(previous_state, state, 0, 0, None, action)
            
            discriminator_reward = self.discriminator.reward(
                [torch.from_numpy(s).float() for s in self.previous_states],
                [torch.from_numpy(a).float() for a in self.actions],
                [torch.from_numpy(s).float() for s in self.states])

            particle = self.particles[self.__current_env]
            # Parameters are previous step, values are the state. The log probs are log_probs, the reward is the discriminator reward.
            particle.store_outcome(self.parameters.numpy(), particle.values.numpy(), log_probs, float(discriminator_reward), False)
            particle.update_policy()
            particle.clear_history()

            # Change the current env to train on another one.
            self.__update_current_env_id()

            # Load the new particle
            new_particle = self.particles[self.__current_env]
            self.parameters = particle.values
            new_particle.update_values()

            # Modify the env parameters for next episode
            self.training_env.envs[0].set_parameters(self.parameters)

            # At the very end of an episode, remove the stored data
            self.states.clear()
            self.previous_states.clear()
            self.actions.clear()
            self.rd_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_reward = 0
            self.current_episode_length = 0
        
            print(f"\r {len(self.episode_lengths)} episodes completed.", end='')

        return True
        
    def _on_training_end(self):
        """
        At the end of the training, we save the metrics.
        """
        print("h")

        with open(f"{self.output_folder}/ref_rewards.json",'w') as f:
            json.dump(self.ref_rewards, f)
        
        with open(f"{self.output_folder}/rd_rewards.json",'w') as f:
            json.dump(self.rd_rewards, f)

        with open(f"{self.output_folder}/episode_lengths.json",'w') as f:
            json.dump(self.episode_lengths, f)