from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import json
import gym

class TrainTestCallback(BaseCallback):

    def __init__(
        self,
        test_env_path: str,
        model: PPO,
        test_every: int,
        output_folder: str,

        verbose: int = 0
    ):
        super().__init__(verbose)

        self.test_env = gym.make(test_env_path)
        self.output_folder = output_folder
        self.test_every = test_every

        self.init_callback(model)

    def _init_callback(self) -> None:
        
        self._rolling_number = 0

        self.test_rewards = []
        self.train_rewards = []
        self.episode_lengths = []

        self.current_episode_length = 0
        self.current_reward = 0
    
    def update_rolling_number(self):
        self._rolling_number = (self._rolling_number+1)%self.test_every

    def _on_step(self):
        """Action to be done at each step."""

        # Retrieve the data
        reward = self.locals['rewards'][0] # float
        done = self.locals['dones'][0] # bool

        # Store them
        self.current_reward += reward
        self.current_episode_length += 1

        if done: # If the episode is over.
            # Test the agent every test_every episode.
            if self._rolling_number == self.test_every-1: 

                test_reward, _  = evaluate_policy(self.model, self.test_env, 10)
                self.test_rewards.append(test_reward)

            self.update_rolling_number()

            # At the very end of an episode, remove the stored data
            self.train_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_reward = 0
            self.current_episode_length = 0
    
        return True
        
    def _on_training_end(self):
        """
        At the end of the training, we save the metrics.
        """

        with open(f"{self.output_folder}/test_rewards.json",'w') as f:
            json.dump(self.test_rewards, f)
        
        with open(f"{self.output_folder}/train_rewards.json",'w') as f:
            json.dump(self.train_rewards, f)

        with open(f"{self.output_folder}/episode_lengths.json",'w') as f:
            json.dump(self.episode_lengths, f)