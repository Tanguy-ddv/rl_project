from .custom_hopper import CustomHopper, SOURCE, TARGET
from .random_hoppers import ADR, UDR, NDR, ADRHopper, UDRHopper, GDRHopper

import gym
gym.envs.register(
    id=UDR,
    entry_point="%s:UDRHopper" % __name__,
    max_episode_steps=500,
)

gym.envs.register(
    id=NDR,
    entry_point="%s:GDRHopper" % __name__,
    max_episode_steps=500,
)

gym.envs.register(
    id=ADR,
    entry_point="%s:ADRHopper" % __name__,
    max_episode_steps=500,
)

gym.envs.register(
        id=SOURCE,
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id=TARGET,
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)


__all__ = ['CustomHopper', 'SOURCE', 'TARGET', 'ADR', 'UDR', 'NDR']
