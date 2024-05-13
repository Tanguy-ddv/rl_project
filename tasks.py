from session import Session
import matplotlib.pyplot as plt
from numpy import arange

def train_with_defined_moving_baseline():
    output_folder = "outputs/defined_moving_baseline"
    session = Session("CustomHopper-source-v0", output_folder, 1, 'cpu')
    session.load_reinforce_with_baseline(None, baseline=0)
    session.train_agent_with_defined_moving_baseline(n_episodes_per_baseline=5000, baselines=[0, 10, 20, 50, 100, 200, 500])
    session.store_infos("baselines = [0, 20, 50, 100, 200, 500], 5000 ep per baseline")

def train_with_increasing_goal_baseline():
    output_folder = "outputs/increasing_goal_baseline2"
    session = Session("CustomHopper-source-v0", output_folder, 0, 'cpu')
    session.load_reinforce_with_baseline(None, baseline=100)
    session.train_agent_with_increasing_goal_baseline(1500, 14, 0)
    session.store_infos("increasing baseline. 1500 ep/step, 14 steps, initial baseline=0")

def compare_baselines():
    output_folder = "outputs/several_baselines"
    session = Session("CustomHopper-source-v0", output_folder, 1, 'cpu')
    for baseline in [0, 10, 20, 50, 100, 200, 500]:
        session.load_reinforce_with_baseline(None, baseline=baseline)
        session.train_agent(n_episode=10000)
    session.store_infos("baselines = [0, 20, 50, 100, 200, 500]")

def compare_lr():
    output_folder = "outputs/several_lr"
    session =  Session('CustomHopper-source-v0', output_folder, 0, 'cpu')
    for lr in [1e-1, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]:
        for baseline in [0, 20, 50]:
            session.load_reinforce_with_baseline(None, lr, baseline)
            session.train_agent(n_episode=5000)
    session.store_infos("lr=[1e-1, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4], baselines = [0, 20, 50]")

def train_actor_critic():
    output_folder = "outputs/actorcritic_basic"
    session = Session('CustomHopper-source-v0', output_folder, verbose=10)
    session.load_actor_critic(None, None)
    step = session.get_step()
    session.train_agent(50000, 1000)
    session.store_infos(f"Step {step}: Actor critic, 50k episodes")

if __name__ == "__main__":
    train_actor_critic()