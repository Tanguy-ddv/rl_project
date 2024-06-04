from .policy import Policy as Actor
from .agent import Agent

import torch
import torch.nn.functional as F

class Critic(torch.nn.Module):

    def __init__(self, state_space: int, hidden=64):
        super().__init__()
        self.state_space = state_space

        self.fc1 = torch.nn.Linear(state_space, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.fc3 = torch.nn.Linear(hidden, 1)

        self._init_weights()
    
    def forward(self, state):
        x = F.tanh(self.fc1(state))
        x = F.tanh(self.fc2(x))
        value_function = self.fc3(x)
        return value_function

    def _init_weights(self):
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.kaiming_normal_(self.fc3.weight)

class ActorCriticAgent(Agent):

    def __init__(self, actor: Actor, critic: Critic, device='cpu', lr_actor: float = 1e-3, lr_critic: float=1e-3):
        super().__init__(actor, device, lr_actor)

        self.critic = critic
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    def update_policy(self):
        """Update the policy at the end of an episode based on the reward obtained during it."""

        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.device).squeeze(-1)
        dones = torch.Tensor(self.done).to(self.device)

        # Compute the values
        with torch.no_grad():
            next_state_values = self.critic(next_states).squeeze(-1)
            target_values = (rewards + (1 - dones)*self.gamma*next_state_values).detach()
        state_values = self.critic(states).squeeze(-1)
        advantage = target_values - state_values

        # Compute the losses
        actor_loss = -(action_log_probs * advantage.detach()).mean()
        critic_loss = F.mse_loss(state_values, target_values)

        # Update the actor
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()

        # Update the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Remove the history at the end of the update
        self.clear_history()