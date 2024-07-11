import torch.distributions as dist
import torch
import gym
from env.custom_randomized_hopper import *
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.functional import normalize
from torch import nn
torch.autograd.set_detect_anomaly(True)
import torch.nn.init as init
torch.autograd.set_detect_anomaly(True)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class Actor(torch.nn.Module):
    def __init__(self, state_space, action_space, lstm_hidden_size, pretrained_actor=None):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden_size = 64
        self.lstm_hidden_size = lstm_hidden_size
        self.tanh = torch.nn.Tanh()

        # LSTM layer
        self.lstm = torch.nn.LSTM(input_size=state_space, hidden_size=self.lstm_hidden_size, batch_first=True)
        
        # Actor network
        self.fc1_actor = torch.nn.Linear(lstm_hidden_size, self.hidden_size)
        self.fc2_actor = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden_size, action_space)

        # Learned standard deviation for exploration at training time
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(action_space) + init_sigma)

        self.init_weights()

        if pretrained_actor is not None:
            self.load_pretrained_weights(pretrained_actor)

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                init.orthogonal_(param)
            elif 'bias' in name:
                init.constant_(param, 0)

        torch.nn.init.kaiming_normal_(self.fc1_actor.weight)
        torch.nn.init.kaiming_normal_(self.fc2_actor.weight)
        torch.nn.init.kaiming_normal_(self.fc3_actor_mean.weight)

    def forward(self, state, hidden):
        x = state
        
        # Process sequences with LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Actor forward pass
        x1 = torch.tanh(self.fc1_actor(lstm_out))
        x2 = torch.tanh(self.fc2_actor(x1))
        mean = self.fc3_actor_mean(x2)
        sigma = torch.nn.functional.softplus(self.sigma)
        normal_dist = Normal(mean, sigma)

        return normal_dist, hidden
    
    def load_pretrained_weights(self, pretrained_actor):
        pretrained_state_dict = pretrained_actor
        new_state_dict = self.state_dict()

        for key in pretrained_state_dict.keys():
            new_key = key.replace('fc1_actor', 'fc1_actor').replace('fc2_actor', 'fc2_actor').replace('fc3_actor_mean', 'fc3_actor_mean')
            if new_key in new_state_dict and pretrained_state_dict[key].shape == new_state_dict[new_key].shape:
                new_state_dict[new_key] = pretrained_state_dict[key]
        
        self.load_state_dict(new_state_dict)


class Critic(torch.nn.Module):
    def __init__(self, state_space, action_space, lstm_hidden_size, pretrained_critic=None):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden_size = 64
        self.lstm_hidden_size = lstm_hidden_size
        self.tanh = torch.nn.Tanh()

        # LSTM layer
        self.lstm = torch.nn.LSTM(input_size=state_space, hidden_size=self.lstm_hidden_size, batch_first=True)
        
        # Critic network

        self.fc1_critic = torch.nn.Linear(self.lstm_hidden_size, self.hidden_size)
        self.fc2_critic = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3_critic = torch.nn.Linear(self.hidden_size, 1)

        self.init_weights()

        if pretrained_critic is not None:
            self.load_pretrained_weights(pretrained_critic)

    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.fc1_critic.weight)
        torch.nn.init.kaiming_normal_(self.fc2_critic.weight)
        torch.nn.init.kaiming_normal_(self.fc3_critic.weight)

    def forward(self, state, hidden):
        x = state
        
        # Process sequences with LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Critic forward pass
        v1 = torch.tanh(self.fc1_critic(lstm_out))
        v2 = torch.tanh(self.fc2_critic(v1))
        value = self.fc3_critic(v2)

        return value
    
    def load_pretrained_weights(self, pretrained_critic):
        pretrained_state_dict = pretrained_critic
        new_state_dict = self.state_dict()

        for key in pretrained_state_dict.keys():
            new_key = key.replace('fc1_critic', 'fc1_critic').replace('fc2_critic', 'fc2_critic').replace('fc3_critic_mean', 'fc3_critic_mean')
            if new_key in new_state_dict and pretrained_state_dict[key].shape == new_state_dict[new_key].shape:
                new_state_dict[new_key] = pretrained_state_dict[key]
        
        self.load_state_dict(new_state_dict)



class Agent(object):
    def __init__(self, actor, critic, device='cpu'):
        self.train_device = device
        self.actor = actor.to(self.train_device)
        self.critic = critic.to(self.train_device)
        
        self.optimizer_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
        self.optimizer_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)

        self.lstm_hidden_size_actor = actor.lstm_hidden_size
        self.lstm_hidden_size_critic = critic.lstm_hidden_size

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

    def update_policy_actor_critic(self): 
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).float().to(self.train_device) 

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        hidden_state = None
        with torch.no_grad():
            next_state_values = self.critic(next_states.unsqueeze(0), hidden_state)
            next_state_values = next_state_values.squeeze(0).squeeze(-1)
            target_values = rewards + (1 - done) * self.gamma * next_state_values
            target_values = target_values.detach()

        # Compute advantage terms
        state_values = self.critic(states.unsqueeze(0), hidden_state)
        state_values = state_values.squeeze(0).squeeze(-1)

        advantages = target_values - state_values
        # Compute actor loss
        actor_loss = -(action_log_probs * advantages.detach()).mean()

        # Compute critic loss
        critic_loss = F.mse_loss(state_values, target_values)

        # Update actor network
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # Update critic network
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()



    def get_action(self, state, hidden, evaluation=False, seed=None):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.train_device)
        if hidden is None:
            hidden = (torch.zeros(1, 1, self.actor.lstm_hidden_size).to(self.train_device), 
                            torch.zeros(1, 1, self.actor.lstm_hidden_size).to(self.train_device))

        normal_dist, hidden = self.actor(state_tensor, hidden)
        _ = self.critic(state_tensor, hidden)

        if evaluation:
            return normal_dist.mean,_, hidden
        else:
            action = normal_dist.sample()
            action_log_prob = normal_dist.log_prob(action).sum()
            return action, action_log_prob, hidden
        
    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
