from typing import Literal
from adr import Discriminator, Particle
import gym
from env.custom_hopper import CustomHopper
from agent.train_and_test import perform
import json
from .ACsession import ACSession

class ADRSession(ACSession):

    def __init__(
            self,
            ref_env_path: Literal['CustomHopper-source-v0','CustomHopper-target-v0'],
            source_for_random_env_path: Literal['CustomHopper-source-v0','CustomHopper-target-v0'],
            nenvs: int,
            output_folder: str,
            lr_actor: float = 1e-3,
            lr_critic: float =1e-3,
            lr_discriminator: float=1e-3,
            lr_particle: float=1e-3,
            verbose: int = 0,
            device: Literal['cpu','cuda'] = 'cpu'):
        
        super().__init__(ref_env_path, output_folder, verbose, device)
        self.load_agent(None, None, lr_actor, lr_critic)
        self.nenvs = nenvs

        self.discriminator = Discriminator(self.state_space, self.action_space, hidden=64, lr=lr_discriminator)

        self.mean_values = self.env.get_parameters()[1:]
        self.nparams = self.mean_values.shape[0]

        # Load the particles
        self.nenvs = nenvs
        self.particles: list[Particle] = []
        self.envs: list[CustomHopper] = []
        for _ in range(nenvs):
            self.particles.append(Particle(self.nparams, self.mean_values, hidden=64, lr=lr_particle))
            self.envs.append(gym.make(source_for_random_env_path))
    
    def _save_metrics(self, ref_rewards: list[float], rd_rewards: list[float], episode_lengths: list[float | int], suffix: str, episode_of_best: int = None):
        super()._save_metrics(ref_rewards, episode_lengths, suffix, episode_of_best)
    
        with open(f"{self.output_folder}/step_{self._step}_{suffix}/random_rewards.json",'w') as f:
            json.dump(rd_rewards, f)
    
    def train(self, n_episodes_per_env: int):
        """
        Train the actor-critic agent on ADR environements.
        """

        ref_rewards = []
        rds_rewards = []
        episode_lengths = []
        best_ref_reward = None
        ep_of_best_ref_reward = 0

        # Start of the training loop
        for episode in range(n_episodes_per_env):

            # Test the agent on the ref environement
            ref_reward = 0
            previous_states, actions, states, ref_reward, episode_length, action_probabilities = perform(self.env, self.agent, False, False)

            for (previous_state, state, action_probs, action) in zip(previous_states, states, action_probabilities, actions): 
                # The done and the reward are not used on the training of the discriminator so we put whatever we want
                self.discriminator.store_ref_outcome(previous_state, state, action_probs, None, None, action)

            ref_rewards.append(ref_reward)
            episode_lengths.append(episode_length)

            if best_ref_reward is None or best_ref_reward < ref_reward:
                best_ref_reward = ref_reward
                ep_of_best_ref_reward = episode

            # Test and train on the random envs.

            rd_rewards = []
            for particle, env in zip(self.particles, self.envs):

                parameters = particle.values
                env.set_parameters(parameters)
                _, log_probs = particle.update_values()

                previous_states, actions, states, rd_reward, _, action_probabilities = perform(env, self.agent, True, False)

                for (previous_state, state, action_probs, action) in zip(previous_states, states, action_probabilities, actions):
                    # The done and the reward are not used on the training of the discriminator so we put whatever we want
                    self.discriminator.store_outcome(previous_state, state, action_probs, None, None, action)
                
                # Update the agent at the end of the episode
                self.agent.update_policy()
                self.agent.clear_history()

                # Update the particle
                discriminator_reward = self.discriminator.reward(previous_states, actions, states)
                particle.store_outcome(parameters.detach().numpy(), particle.values.detach().numpy(), log_probs, discriminator_reward, False)
                particle.update_policy()
                particle.clear_history()

                rd_rewards.append(rd_reward)
            
            # Update the discriminator
            self.discriminator.update_policy()
            self.discriminator.clear_history()
            rds_rewards.append(rd_rewards)
        
        self._save_metrics(ref_rewards, rds_rewards, episode_lengths, 'train', ep_of_best_ref_reward)