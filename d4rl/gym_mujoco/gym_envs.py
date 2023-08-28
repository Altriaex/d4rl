from .. import offline_env
#from gymnasium.envs.mujoco import HalfCheetahEnv, AntEnv, HopperEnv, Walker2dEnv
#from gymnasium.envs.mujoco.ant_v5 import AntEnv
#from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv
#from gymnasium.envs.mujoco.hopper_v5 import HopperEnv
#from gymnasium.envs.mujoco.walker2d_v5 import Walker2dEnv
from ..utils.wrappers import NormalizedBoxEnv
import gymnasium as gym

'''class OfflineAntEnv(AntEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        AntEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineHopperEnv(HopperEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        HopperEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineHalfCheetahEnv(HalfCheetahEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        HalfCheetahEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineWalker2dEnv(Walker2dEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        Walker2dEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)'''


def get_ant_env(**kwargs):
    env = gym.make("Ant-v5")
    return NormalizedBoxEnv(offline_env.OfflineEnvWrapper(env, **kwargs))
    #return NormalizedBoxEnv(OfflineAntEnv(**kwargs))

def get_cheetah_env(**kwargs):
    env = gym.make("HalfCheetah-v5")
    return NormalizedBoxEnv(offline_env.OfflineEnvWrapper(env, **kwargs))

def get_hopper_env(**kwargs):
    env = gym.make("Hopper-v5")
    return NormalizedBoxEnv(offline_env.OfflineEnvWrapper(env, **kwargs))

def get_walker_env(**kwargs):
    env = gym.make("Walker2d-v5")
    return NormalizedBoxEnv(offline_env.OfflineEnvWrapper(env, **kwargs))

if __name__ == '__main__':
    """Example usage of these envs"""
    pass
