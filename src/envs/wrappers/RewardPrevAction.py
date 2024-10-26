import sys
sys.path.append('./')
sys.path.append('../')

import jax
import jax.numpy as jnp
from src.envs.wrappers.purerl import GymnaxWrapper, environment, Optional, partial, Tuple, chex, spaces, Union
from flax import struct

## Add previous action and reward to the observation space
class AliasRewardPrevAction(GymnaxWrapper):
    """Adds a reward and the last action."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        action_space = self._env.action_space(params)
        og_observation_space = self._env.observation_space(params)
        if type(action_space) == spaces.Discrete:
            low = jnp.concatenate([og_observation_space.low, jnp.zeros((action_space.n+1,))])
            high = jnp.concatenate([og_observation_space.high, jnp.ones((action_space.n+1,))])
            return spaces.Box(
                low=low,
                high=high,
                shape=(self._env.observation_space(params).shape[-1]+action_space.n+1,), # NOTE: ASSUMES FLAT RIGHT NOW
                dtype=self._env.observation_space(params).dtype,
            )
        elif type(action_space) == spaces.Box:
            low = jnp.concatenate([og_observation_space.low, jnp.array([action_space.low]), jnp.array([0.0])])
            high = jnp.concatenate([og_observation_space.high, jnp.array([action_space.high]), jnp.array([1.0])])
            return spaces.Box(
                low=low,
                high=high,
                shape=(self._env.observation_space(params).shape[-1]+2,), # NOTE: ASSUMES FLAT RIGHT NOW
                dtype=self._env.observation_space(params).dtype,
            )
        else:
            raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        action_space = self._env.action_space(params)
        obs, state = self._env.reset(key, params)
        if isinstance(action_space, spaces.Box):
            obs = jnp.concatenate([obs, jnp.array([0.0, 0.0])])
        elif isinstance(action_space, spaces.Discrete):
            obs = jnp.concatenate([obs, jnp.zeros((action_space.n,)), jnp.array([0.0])])
        else:
            raise NotImplementedError
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
        action_space = self._env.action_space(params)
        if isinstance(action_space, spaces.Discrete):
            # obs = jnp.concatenate([obs, jnp.array([action, 0.0])])
            action_in = jnp.zeros((action_space.n,))
            action_in = action_in.at[action].set(1.0)
            obs = jnp.concatenate([obs, action_in, jnp.array([reward])])
        else:
            obs = jnp.concatenate([obs, action, jnp.array([reward])])
        return obs, state, reward, done, info



