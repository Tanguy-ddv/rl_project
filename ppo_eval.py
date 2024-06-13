import gym
import env
from stable_baselines3 import PPO
from agent.train_and_test import performPPO

def train_ppo(train_env: gym.Env, name: str, total_timesteps, ppo_kwargs = {}, seed = 42):

    model = PPO("MlpPolicy", train_env, verbose=0, **ppo_kwargs, seed=seed)
    model.learn(total_timesteps=total_timesteps, callback=myCallBack(0))
    model.save(name)
    return model


from stable_baselines3.common.callbacks import BaseCallback

class myCallBack(BaseCallback):

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.t = 0
    
    def _on_rollout_start(self) -> None:
        print('yoyo')
    
    def _on_step(self) -> None:
        previous_state = self.locals['obs_tensor'][0].numpy()
        state = self.locals['new_obs'][0]
        action = self.locals['actions'][0]
        reward = self.locals['rewards'][0]
        log_probs = self.locals['log_probs']
        done = self.locals['dones'][0]

        print('previous states', type(previous_state))
        print('state', type(state))
        print('action', type(action))
        print('reward', reward)
        print('log_probs', log_probs)
        print('done', done)

        self.t += 1
        return self.t <= 2

train_ppo(gym.make(env.SOURCE), 'yehe', 100000)