import jax
import jax.numpy as jnp
import chex
import numpy as np
from functools import partial
from gymnax.environments import environment, spaces
from typing import Optional, Tuple, Union
from gymnax.wrappers.purerl import GymnaxWrapper


class CartPolePWrapper(GymnaxWrapper):
    """CartPole with masked positons"""

    def __init__(self, env: environment.Environment):
        super().__init__(env)
        self.obs_idx = [0, 2]

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        return spaces.Box(
            low=self._env.observation_space(params).low,
            high=self._env.observation_space(params).high,
            shape=(np.prod((len(self.obs_idx)),),),
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        obs = jnp.reshape(obs, (-1,))
        obs = jnp.take(obs,jnp.array(self.obs_idx))
        
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(
            key, state, action, params
        )
        obs = jnp.reshape(obs, (-1,))
        obs = jnp.take(obs,jnp.array(self.obs_idx))
        return obs, state, reward, done, info