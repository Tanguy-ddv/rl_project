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

class ParticleNN(nn.Module):

    def __init__(self, nparams: int, mean_values: np.ndarray, hidden=64):
        """
        nparams: the number of parameter to randomize
        mean_value: an array of dimension (1, nparams) that represent the average value for the parameters.
        """
        super(nn.Module, self).__init__()

        self.mean_values = torch.tensor(mean_values)

        self.l1 = nn.Linear(nparams, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, nparams)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        # Clip the value between - mean_values and + mean_values
        x = F.relu(self.l3(x))/(np.pi/2)*self.mean_values
        return x

class Particle:

    def __init__(self, nparams: int, mean_values: np.ndarray, hidden=64, lr=1e-3):
        self.network = ParticleNN(nparams, mean_values, hidden)
        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=lr)

        # Initialize the values
        self.values = torch.distributions.Uniform(torch.zeros_like(self.network.mean_values), self.network.mean_values*2)
    
    def update_values(self):

        self.values += self.network(self.values)
        self.values = torch.clip(self.values, torch.zeros_like(self.network.mean_values), self.network.mean_values*2)
    
    def create_env(self):
        """Register and create a new environment with the current values."""
        # TODO
    
    def train():
        """Train the particle."""
        # TODO