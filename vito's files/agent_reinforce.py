import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

torch.autograd.set_detect_anomaly(True)


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        return normal_dist


class Agent(object):
    def __init__(self, policy, device='cpu', update_frequency=10):
        self.update_frequency = update_frequency
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        
        self.optimizer_actor = torch.optim.Adam(policy.parameters(), lr=1e-3)
        self.optimizer_critic = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.values = []


    def update_policy_reinforce(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        
        # baseline(eventually)
        b = 20

        #   - compute discounted returns
        #discounted_rewards = discount_rewards(rewards, self.gamma)
        discounted_rewards = discount_rewards(rewards, self.gamma) - b #eventually for the baseline trial
        
        #   - compute policy gradient loss function given actions and returns
        loss = []
        for R, log_prob in zip(discounted_rewards,  action_log_probs):
            loss.append(-R * log_prob *self.gamma)
        
        loss = torch.stack(loss, dim=0).sum()

        #   - compute gradients nd step the optimaizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #

        #wE CAN IMPLEMENT A FUNCTION THAT DOES THIS AND CALL IT HERE
        # Clear collected experiences
        self.states.clear()
        self.next_states.clear()
        self.action_log_probs.clear()
        self.rewards.clear()
        self.done.clear()



    def update_policy_actor_critic(self):

        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)
        values = torch.stack(self.values, dim=0).to(self.train_device).squeeze(-1)

        
        #   - compute boostrapped discounted return estimates (just for Reinforce)
        discounted_rewards = discount_rewards(rewards, self.gamma)

        #   - compute advantage terms
        actor_loss = []
        critic_loss = []
        i = 0
        for log_prob, value, dis_rew, rew, next_state in zip(action_log_probs, values, discounted_rewards, rewards, next_states):
        
            if i < len(states)-1:  #not in terminal state   check Done, insteaf
                _ , value_next_state = self.policy(next_state) 
                rew_bootstraped = rew + self.gamma*value_next_state
                advantage = dis_rew - rew_bootstraped - value
            else: #terminal state
                rew_bootstraped = rew
                advantage = dis_rew - rew_bootstraped - value 


            #   - compute actor loss and critic loss    
            actor_loss.append(-log_prob * advantage)
            critic_loss.append(F.mse_loss(value.squeeze(), rew_bootstraped.clone().detach().unsqueeze(0)))
            i +=1

        # compute gradients and step the optimizer
        self.optimizer_actor.zero_grad()
        actor_loss = torch.stack(actor_loss, dim=0).sum()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        critic_loss = torch.stack(critic_loss, dim=0).sum()
        critic_loss.requires_grad = True
        critic_loss.backward()
        self.optimizer_critic.step()


        #WE CAN IMPLEMENT A FUNCTION THAT DOES THIS AND CALL IT HERE
        # Clear collected experiences
        self.states.clear()
        self.next_states.clear()
        self.action_log_probs.clear()
        self.rewards.clear()
        self.done.clear()
        self.values.clear()
        #

        return        


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
 

