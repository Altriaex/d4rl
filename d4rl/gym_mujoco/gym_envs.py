from mailbox import NotEmptyError
from .. import offline_env

from ..utils.wrappers import NormalizedBoxEnv
import gymnasium as gym

def get_ant_env(**kwargs):
    #env = gym.make("Ant-v5", use_contact_forces=True)
    #return NormalizedBoxEnv(offline_env.OfflineEnvWrapper(env, **kwargs))
    raise NotEmptyError("There is a contact forces issue in the ant datasets of D4RL. See https://gymnasium.farama.org/environments/mujoco/ant/ for details.")

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
