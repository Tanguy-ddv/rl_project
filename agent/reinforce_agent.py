from .policy import Policy
from .agent import Agent, discount_rewards

import torch

class ReinforceAgent(Agent):

    def __init__(self, policy: Policy, device='cpu', lr: float = 1e-3, baseline:int = 0):
        super().__init__(policy, device, lr)
        self.baseline = baseline

    def update_policy(self):
        """Update the policy at the end of an episode based on the reward obtained during it."""
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.device).squeeze(-1)

        discounted_r = discount_rewards(rewards, self.gamma) - self.baseline
        loss = []
        for R, log_prob in zip(discounted_r, action_log_probs):
            loss.append(-R*log_prob*self.gamma)

        loss = torch.stack(loss, dim=0).sum()

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
