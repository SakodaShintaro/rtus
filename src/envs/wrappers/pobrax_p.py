import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any
from gymnax.environments import environment, spaces
from brax import envs
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper

           
class POBraxAntPWrapper:
    def __init__(self, backend="positional"):
        env = envs.get_environment(env_name='ant', backend=backend)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = 13 # Only Positional information

    def reset(self, key, params=None):
        state = self._env.reset(key)
        return state.obs[:self.observation_size], state

    def step(self, key, state, action, params=None):
        next_state = self._env.step(state, action)
        return next_state.obs[:self.observation_size], next_state, next_state.reward, next_state.done > 0.5, {}

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.observation_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )

class POBraxCheetahPWrapper:
    def __init__(self, backend="positional"):
        env = envs.get_environment(env_name='halfcheetah', backend=backend)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = 9 # Only Positional information

    def reset(self, key, params=None):
        state = self._env.reset(key)
        return state.obs[:self.observation_size], state

    def step(self, key, state, action, params=None):
        next_state = self._env.step(state, action)
        return next_state.obs[:self.observation_size], next_state, next_state.reward, next_state.done > 0.5, {}

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.observation_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )        

class POBraxHopperPWrapper:
    def __init__(self, backend="positional"):
        env = envs.get_environment(env_name='hopper', backend=backend)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = 5 # Only Positional information

    def reset(self, key, params=None):
        state = self._env.reset(key)
        return state.obs[:self.observation_size], state

    def step(self, key, state, action, params=None):
        next_state = self._env.step(state, action)
        return next_state.obs[:self.observation_size], next_state, next_state.reward, next_state.done > 0.5, {}

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.observation_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )

class POBraxWalkerPWrapper:
    def __init__(self, backend="positional"):
        env = envs.get_environment(env_name='walker2d', backend=backend)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = 8 # Only Positional information

    def reset(self, key, params=None):
        state = self._env.reset(key)
        return state.obs[:self.observation_size], state

    def step(self, key, state, action, params=None):
        next_state = self._env.step(state, action)
        return next_state.obs[:self.observation_size], next_state, next_state.reward, next_state.done > 0.5, {}

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.observation_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )