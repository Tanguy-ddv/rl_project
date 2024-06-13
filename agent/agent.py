
from .policy import Policy
from abc import abstractmethod, ABC
import torch

def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class Agent(ABC):

    def __init__(self, policy: Policy, device='cpu', lr: float = 1e-3, gamma: float=0.99):
        super().__init__()
        self.device = device
        self.policy = policy.to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma

        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.actions = []
        self.rewards = []
        self.done = []
    
    @abstractmethod
    def update_policy(self):
        """Update the policy at the end of an episode based on the reward obtained during it."""

    def clear_history(self):
        """Clear the history of the previous actions"""
        self.states.clear()
        self.next_states.clear()
        self.action_log_probs.clear()
        self.rewards.clear()
        self.done.clear()
        self.actions = []

    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.device)

        normal_dist: torch.distributions.Normal = self.policy(x)
        # In the case of different policies that directly return an action and not a distribution

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done, action=None):
        """Save the outcome on the history"""
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
        if action is not None:
            self.actions.append(torch.from_numpy(action).float())

