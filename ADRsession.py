"""
The ADR session contains all the functions needed to compute an ADR training on the hopper.
"""

from adr.particle import Particle
from adr import hopper_for_adr # To have the adr registered.
from env import custom_hopper # To have the target registered
from adr.discriminator import Discriminator
import gym
from agent.actor_critic_agent import ActorCriticAgent, Actor, Critic

class ADRSession:

    def __init__(self, nenvs, hidden=64, lr=1e-3) -> None:

        self.ref_env: custom_hopper.CustomHopper = gym.make("CustomHopper-target-v0")

        self.state_space = self.ref_env.observation_space.shape[-1]
        self.action_space = self.ref_env.action_space.shape[-1]
        self.mean_values = self.ref_env.get_parameters()[1:]
        self.nparams = self.mean_values.shape[0]

        # Load the particles
        self.discriminator = Discriminator(self.state_space, self.action_space, hidden, lr)

        # Load the actor-critic agent
        actor = Actor(self.state_space, self.action_space, hidden)
        critic = Critic(self.state_space, hidden)
        self.agent = ActorCriticAgent(actor, critic, lr_actor=lr, lr_critic=lr)

        # Load the particles
        self.nenvs = nenvs
        self.particles: list[Particle] = []
        self.envs: list[hopper_for_adr.HopperForADR] = []
        for _ in range(nenvs):
            self.particles.append(Particle(self.nparams, self.mean_values, hidden, lr))
            self.envs.append(gym.make("ADRHopper-v0"))
    
    def train(self, n_episodes: int):
        """
        Train the actor-critic agent on ADR environements.
        """

        # Start of the training loop
        for episode in range(n_episodes):

            # Test the agent on the ref environement
            done = False
            state = self.ref_env.reset()
            while not done:

                action, action_probabilities = self.agent.get_action(state)
                previous_state = state

                state, reward, done, _info = self.ref_env.step(action.detach().cpu().numpy())
                self.agent.store_outcome(previous_state, state, action_probabilities, reward, done)

                self.discriminator.store_ref_outcome(previous_state, state, action_probabilities, reward, done, action)

            # Test and train on the reference envs.

            for particle, env in zip(self.particles, self.envs):

                state = env.reset()
                parameters = particle.values
                env.set_parameters(parameters)
                _, log_probs = particle.update_values()

                states, actions, next_states = [], [], []
                # Reset the training data
                done = False
                while not done:  # Loop until the episode is over

                    action, action_probabilities = self.agent.get_action(state)
                    previous_state = state

                    state, reward, done, _info = env.step(action.detach().cpu().numpy())
                    self.agent.store_outcome(previous_state, state, action_probabilities, reward, done)
                    self.discriminator.store_outcome(previous_state, state, action_probabilities, reward, done, action)
                    
                    states.append(previous_state)
                    actions.append(action)
                    next_states.append(state)

                # Update the agent at the end of the episode
                self.agent.update_policy()
                self.agent.clear_history()

                # Update the particle
                discriminator_reward = self.discriminator.reward(states, actions, next_states)
                particle.store_outcome(parameters, particle.values, log_probs, discriminator_reward, False)
                particle.update_policy()
                particle.clear_history()
            
            # Update the discriminator
            self.discriminator.update_policy()
            self.discriminator.clear_history()
        
        

        
if __name__ == '__main__':
    s = ADRSession(2)
    s.train(2)