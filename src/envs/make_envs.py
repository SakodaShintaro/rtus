import sys
sys.path.append('./')
sys.path.append('../')

from src.envs.wrappers.purerl import *
import gymnax
from typing import Any, Callable
from src.configs.EnvConfig import EnvConfig
from src.envs.wrappers.pobrax_p import *
from src.envs.wrappers.pobrax_v import *
from src.envs.wrappers.gymnax_p import * 
from src.envs.wrappers.RewardPrevAction import *
from src.envs.popjaxrl.registration import make as popjax_make


pobrax_p_envs = {
    'ant_p': POBraxAntPWrapper,
    'cheetah_p': POBraxCheetahPWrapper,
    'hopper_p': POBraxHopperPWrapper,
    'walker_p': POBraxWalkerPWrapper,
}

pobrax_v_envs = {
    'ant_v': POBraxAntVWrapper,
    'cheetah_v': POBraxCheetahVWrapper,
    'hopper_v': POBraxHopperVWrapper,
    'walker_v': POBraxWalkerVWrapper,
}
  
gymnax_p_envs = {'CartPole-v1': CartPolePWrapper,
                 }

    
def make_brax(env_name):
    env, env_params = BraxGymnaxWrapper(env_name), None
    return env, env_params

def make_gymnax(env_name):
    env, env_params = gymnax.make(env_name)
    return env, env_params

def make_pobrax_pos(env_name):
    env, env_params = pobrax_p_envs[env_name](),None
    return env, env_params

def make_pobrax_vel(env_name):
    env, env_params = pobrax_v_envs[env_name](),None
    return env, env_params

def make_gymnax_p(env_name):
    env, env_params = gymnax.make(env_name)
    env = gymnax_p_envs[env_name](env)
    return env, env_params

def make_popjax(env_name):
    env, env_params = popjax_make(env_name)
    return env, env_params

envs = {'brax': make_brax,
        'pobrax_p': make_pobrax_pos,
        'pobrax_v': make_pobrax_vel,
        'gymnax': make_gymnax,
        'gymnax_p': make_gymnax_p,
        'popjax': make_popjax,}

def make_env(env_config: EnvConfig,gamma:float=0.99):
    # gamma is needed for reward normalization wrapper
    env, env_params = envs[env_config.domain](env_config.env_name)
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    if env_config.clip_action and env_config.continous_action:
        env = ClipAction(env,low=-2,high=2)

    if env_config.normalize_obs:
        env = NormalizeVecObservation(env)

    if env_config.normalize_reward:
        env = NormalizeVecReward(env,gamma)
    
    if env_config.add_reward_prev_action:
        env = AliasRewardPrevAction(env)
    return env, env_params
