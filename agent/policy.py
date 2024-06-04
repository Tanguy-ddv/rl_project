import torch
import torch.nn.functional as F
from torch.distributions import Normal


class Policy(torch.nn.Module):
    """The policy of an agent."""
    
    def __init__(self, state_space: int, action_space: int, hidden: int = 64):
        super().__init__()

        self.fc1 = torch.nn.Linear(state_space, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.fc3 = torch.nn.Linear(hidden, action_space)

        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(action_space)+init_sigma)

        self._init_weights()


    def _init_weights(self):
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.kaiming_normal_(self.fc3.weight)


    def forward(self, state) -> Normal:

        x = F.tanh(self.fc1(state))
        x = F.tanh(self.fc2(x))
        action_mean = self.fc3(x)

        sigma = self.sigma_activation(self.sigma)
        action_dist = Normal(action_mean, sigma)

        return action_dist