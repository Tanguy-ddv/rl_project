"""
The discriminator is a NN that takes as input trajectories and 
outputs a boolean: if the trajectory is from the reference env or from a randomized one.

Input: a trajectory is a S-A-S' tensor
Output: A probability in [0,1] to be from the reference environments.

Training: Use generated trajectories to train:
    - the reference traj is obtained in the target env
    - the randomized traj are obtained in the randomized env.
    To train, map the ref to 0 and the randomized to 1.
    With a list of S-A-S' (a whole episode), map every episode to a probability, then take the mean.
"""

from torch import nn
import torch.nn.functional as F
import torch

def separate_trajectory(
        states: list[torch.Tensor],
        actions: list[torch.Tensor],
        next_states: list[torch.Tensor]
):
    trajs = []
    for state, action, next_state in zip(states, actions, next_states):
        trajs.append(torch.concatenate((state, action, next_state)))
    return torch.concatenate(trajs, dim=1)

class DiscriminatorNN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden=64):
        super(nn.Module, self).__init__()
        input_dim = state_dim + action_dim + state_dim
        self.l1 = nn.Linear(input_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        # Clip in [0,1]
        x = F.sigmoid(self.l3(x))
        return x
    
class Discriminator():

    def __init__(self, state_dim, action_dim, hidden=64, lr=1e-3):
        self.network = DiscriminatorNN(state_dim, action_dim, hidden=hidden)
        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=lr)
        self.criterion = nn.BCELoss()

    def reward(self, states: list[torch.Tensor], actions: list[torch.Tensor], next_states: list[torch.Tensor]):
        probs = self.network(separate_trajectory(states, actions, next_states))
        return torch.log(probs.mean()+1e-8)
    
    def train(
        self, 
        rd_states: list[torch.Tensor], 
        rd_actions: list[torch.Tensor], 
        rd_next_states: list[torch.Tensor],
        ref_states: list[torch.Tensor], 
        ref_actions: list[torch.Tensor], 
        ref_next_states: list[torch.Tensor]
    ):

        rd_probs = self.network(separate_trajectory(rd_states, rd_actions, rd_next_states))
        ref_probs = self.network(separate_trajectory(ref_states, ref_actions, ref_next_states))

        self.optimizer.zero_grad()

        loss = self.criterion(rd_probs, torch.ones_like(rd_probs)) + self.criterion(ref_probs, torch.zeros_like(ref_probs))
        loss.backward()

        self.optimizer.step()
