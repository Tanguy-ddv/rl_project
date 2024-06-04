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
from agent.agent import Agent
import torch

def separate_trajectory(
        states: list[torch.Tensor],
        actions: list[torch.Tensor],
        next_states: list[torch.Tensor]
):
    trajs = []
    for state, action, next_state in zip(states, actions, next_states):
        trajs.append(torch.concatenate((state, action, next_state)))
    return torch.stack(trajs)

class DiscriminatorNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        input_dim = state_dim + action_dim + state_dim
        self.l1 = nn.Linear(input_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)
    
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.kaiming_normal_(self.l1.weight)
        torch.nn.init.kaiming_normal_(self.l2.weight)
        torch.nn.init.kaiming_normal_(self.l3.weight)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        # Clip in [0,1]
        x = F.sigmoid(self.l3(x))
        return x
    
class Discriminator(Agent):

    def __init__(self, state_dim, action_dim, hidden=64, lr=1e-3):
        super().__init__(DiscriminatorNetwork(state_dim, action_dim, hidden=hidden), lr=lr)

        self.ref_states = []
        self.ref_next_states = []
        self.ref_action_log_probs = []
        self.ref_rewards = []
        self.ref_done = []
        self.ref_actions = []

    def reward(self, states: list[torch.Tensor], actions: list[torch.Tensor], next_states: list[torch.Tensor]):
        probs = self.policy(separate_trajectory(states, actions, next_states))
        return torch.log(probs.mean()+1e-8)
    
    def update_policy(self):

        rd_probs = self.policy(separate_trajectory(self.states, self.actions, self.next_states))
        ref_probs = self.policy(separate_trajectory(self.ref_states, self.ref_actions, self.ref_next_states))

        self.policy_optimizer.zero_grad()

        loss = F.binary_cross_entropy(rd_probs, torch.ones_like(rd_probs))
        + F.binary_cross_entropy(ref_probs, torch.zeros_like(ref_probs))

        loss.backward()

        self.policy_optimizer.step()
    
    def clear_history(self):
        """Clear the history of the previous actions"""
        # Clear the trajectory generated in the random.
        self.states.clear()
        self.next_states.clear()
        self.action_log_probs.clear()
        self.rewards.clear()
        self.done.clear()
        self.actions.clear()

        # Clear the trajectory generated in the reference.
        self.ref_states.clear()
        self.ref_next_states.clear()
        self.ref_action_log_probs.clear()
        self.ref_rewards.clear()
        self.ref_done.clear()
        self.ref_actions.clear()

    def store_ref_outcome(self, state, next_state, action_log_prob, reward, done, action):
        """Save the outcome on the history"""
        self.ref_states.append(torch.from_numpy(state).float())
        self.ref_next_states.append(torch.from_numpy(next_state).float())
        self.ref_action_log_probs.append(action_log_prob)
        self.ref_rewards.append(torch.Tensor([reward]))
        self.ref_done.append(done)
        self.ref_actions.append(torch.from_numpy(action).float())