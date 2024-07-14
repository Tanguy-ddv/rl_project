from .policy import Policy
from .agent import Agent, discount_rewards

import torch

class ReinforceAgent(Agent):

    def __init__(self, policy: Policy, device='cpu', lr: float = 1e-3, baseline:int = 0):
        super().__init__(policy, device, lr)
        self.baseline = baseline

    def update_policy(self):
        """Update the policy at the end of an episode based on the reward obtained during it."""
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        rewards=self.rewards
        discounted_returns = []
        dr = 0
        for r in rewards[::-1]:
            dr = dr*self.gamma + r
            discounted_returns.append(dr)
        discounted_returns.reverse()
        discounted_returns = torch.stack(discounted_returns, dim=0).to(self.train_device).squeeze(-1)
        discounted_returns = (discounted_returns - discounted_returns.mean())/ discounted_returns.std()
        discounted_returns = discounted_returns - self.baseline
        self.policy_optimizer.zero_grad()
        loss = -torch.mul(discounted_returns, action_log_probs).mean()
        loss.backward()
        self.policy_optimizer.step()
        self.clear_history()
        return