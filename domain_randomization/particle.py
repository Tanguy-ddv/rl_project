"""
The particles are the functions that generate an environment. They are torch.nn.module subclass.

The input of the particle is the current randomized parameters.
The output of the particle is the translation that should be applied on these parameters.
The particle is update with the discriminator reward, calculated via the discriminator class.
"""

from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from agent.actor_critic_agent import ActorCriticAgent, Critic

class ParticleNN(nn.Module):

    def __init__(self, nparams: int, mean_values: np.ndarray, hidden=64):
        """
        nparams: the number of parameter to randomize
        mean_value: an array of dimension (1, nparams) that represent the average value for the parameters.
        """
        super().__init__()

        self.mean_values = torch.tensor(mean_values)

        self.l1 = nn.Linear(nparams, hidden)
        self.l3 = nn.Linear(hidden, nparams)

        self.sigma_activation = F.softplus
        init_sigma = mean_values.mean()/20
        self.sigma = torch.nn.Parameter(torch.zeros(nparams)+init_sigma)

        self._init_weights()
    
    def _init_weights(self):
        torch.nn.init.kaiming_normal_(self.l1.weight)
        torch.nn.init.kaiming_normal_(self.l3.weight)

    
    def forward(self, x):
        x = F.relu(self.l1(x))
        # Clip the value between - mean_values and + mean_values
        x = F.tanh(self.l3(x))/(np.pi/2)*self.mean_values

        sigma = self.sigma_activation(self.sigma)
        delta_dist = torch.distributions.Normal(x, sigma)

        return delta_dist

class Particle(ActorCriticAgent):

    def __init__(self, nparams: int, mean_values: np.ndarray, hidden=64, lr=1e-3):
        super().__init__(
            ParticleNN(nparams, mean_values, hidden),
            Critic(nparams, hidden)
        )

        # Initialize the values
        self.values = torch.distributions.Uniform(torch.zeros_like(self.policy.mean_values), self.policy.mean_values*2).sample()

    def update_values(self):

        delta, log_probs = self.get_action(self.values.detach().numpy())
        self.values += delta
        self.values = torch.clip(self.values, torch.zeros_like(self.policy.mean_values), self.policy.mean_values*2)

        return delta, log_probs