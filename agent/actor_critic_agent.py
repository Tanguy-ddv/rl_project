from .policy import Policy
from .agent import Agent, discount_rewards

import numpy as np
import torch

class Critic(torch.nn.Module):

    def __init__(self, state_space: int):
        super().__init__()
        self.state_space = state_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic = torch.nn.Linear(self.hidden, 1)

        self.init_weights()
    
    def forward(self, x):
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        value_function = self.fc3_critic(x_critic)
        return value_function

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

class ActorCriticAgent(Agent):
    def __init__(self, policy: Policy, critic: Critic, device='cpu', lr_policy: float = 1e-3, lr_critic: float=1e-3):
        super().__init__(policy, device, lr_policy)

        self.critic = critic
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    def update_policy(self):
        """Update the policy at the end of an episode based on the reward obtained during it."""

        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.device).squeeze(-1)
        dones = torch.Tensor(self.done).to(self.device)

        loss_policy = []
        # Compute the loss for the policy and the value function
        for R, log_prob, state, next_state, done in zip(rewards, action_log_probs, states, next_states, dones):
            predicted_value = self.critic(state)
            target_value = R + (self.gamma*self.critic(next_state))*int(not done)
            loss_policy.append(-log_prob*(target_value - predicted_value).detach())

        # Update the policy
        loss_policy = torch.stack(loss_policy, dim=0).sum()
        self.policy_optimizer.zero_grad()
        loss_policy.backward()
        self.policy_optimizer.step()

        # update the value function
        loss_critic = torch.nn.MSELoss()(predicted_value, target_value)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()