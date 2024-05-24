"""Implementation of the Hopper environment supporting
domain randomization optimization.
"""
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv
from typing import Literal


class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self,
                 domain: Literal['uniform', 'normal', 'source', 'target', None] = None,
                 **kwargs):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses
        self.done = False
        self.domain = domain

        if domain != 'target':  # Source environment has an imprecise torso mass (1kg shift)
            self.sim.model.body_mass[1] -= 1.0
        
        self.torso_mass = deepcopy(self.sim.model.body_mass[1])

        if (('upper' in kwargs and (not 'lower' in kwargs))
            or ('upper' in kwargs and 'max_prop' in kwargs)
            or ('lower' in kwargs and 'max_prop' in kwargs)
        ): # We don't want them together
            raise KeyError("The specified parameters are wrong")

        if 'upper' in kwargs and self.domain == 'uniform':
            self.upper = kwargs['upper']
            self.mass_max = np.ones_like(self.original_masses)*kwargs['upper']

        if 'lower' in kwargs and self.domain == 'uniform':
            self.mass_min = np.ones_like(self.original_masses)*kwargs['lower']

        if 'max_prop' in kwargs and self.domain == 'uniform':
            self.mass_max = kwargs['max_prop'] * self.original_masses
            self.mass_min = np.zeros_like(self.mass_max)

        if 'std_dev_prop' in kwargs and self.domain == 'normal':
            self.std_dev = kwargs['std_dev_prop']*self.original_masses
        
    def reset_parameters(self):
        """Reset the masses, with a randomization or not."""
        if self.domain == 'normal':
            self.set_parameters(self.sample_normal_parameters())
        elif self.domain == 'uniform':
            self.set_parameters(self.sample_uniform_parameters())
        else:
            pass
        self.sim.model.body_mass[1] = self.torso_mass # assure the torso mass never change.

    def sample_uniform_parameters(self):
        # Sample new masses from a uniform distribution
        new_masses = np.random.uniform(self.mass_min, self.mass_max)
        return new_masses

    def sample_normal_parameters(self):
        # Sample new masses from a normal distribution
        new_masses = np.random.normal(loc=self.original_masses, scale=self.mass_std_dev)
        # Ensure masses are not negative
        new_masses = np.clip(new_masses, a_min=0, a_max=None)
        return new_masses

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array(self.sim.model.body_mass[1:])
        return masses

    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[1:] = task

    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}


    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])


    def reset_model(self):
        """Reset the environment to a random initial state"""

        # Sample new parameters at the start of each episode
        self.reset_parameters()

        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


    def set_mujoco_state(self, state):
        """Set the simulator to a specific state

        Parameters:
        ----------
        state: ndarray,
               desired state
        """
        mjstate = deepcopy(self.get_mujoco_state())

        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]

        self.set_sim_state(mjstate)


    def set_sim_state(self, mjstate):
        """Set internal mujoco state"""
        return self.sim.set_state(mjstate)


    def get_mujoco_state(self):
        """Returns current mjstate"""
        return self.sim.get_state()



"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)

gym.envs.register(
    id="CustomHopper-normal-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "uniform"}
)

gym.envs.register(
    id="CustomHopper-uniform-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "normal"}
)

def register_uniform(lower_bound, upper_bound, name):
    gym.envs.register(
    id=name,
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "uniform", "lower": lower_bound, "upper" : upper_bound}
)
