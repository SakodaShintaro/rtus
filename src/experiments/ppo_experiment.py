import sys
sys.path.append('./')
sys.path.append('../')

import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Sequence, NamedTuple, Any, Callable, Optional, Tuple

from src.envs.wrappers.purerl import *
from src.agents.ActorCritic import ActorCritic
from src.configs.PPOConfig import PPOConfig
from src.configs.EnvConfig import EnvConfig
from src.configs.ExpConfig import ExpConfig
from src.envs.wrappers.purerl import FlattenObservationWrapper, LogWrapper
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
import wandb 
import argparse
import yaml 

from src.envs.make_envs import *

from src.algorithms.PPO import AgentState, Transition, init_agent_state, wandb_ppo_logging,update_step,init_agent_state,agent_step



 
@partial(jax.jit, static_argnums=(2,))
def exp_step(runner_state, unused,env):
    
    @jax.jit
    def interacton_step(interaction_state, unused):
        agent_state,env_state,env_params,last_obs, rng = interaction_state
        # agent step 
        rng, _rng = jax.random.split(rng)
        action, value, log_prob = agent_step(agent_state, jnp.expand_dims(last_obs,(0,1)), _rng)
        # env step
        rng, rng_step = jax.random.split(rng)
        obs, env_state, reward, done, info = env.step(rng_step, env_state, action.squeeze(), env_params)
        # save transition
        transition = Transition(last_obs,action.squeeze(),reward,done, value.squeeze(),log_prob.squeeze(),info)
        interaction_state = (agent_state, env_state,env_params, obs, rng)
        return interaction_state, transition
    
    # Collect a trajectory
    rollout_steps = runner_state[0].ppo_config.rollout_steps
    runner_state, traj_batch = jax.lax.scan(interacton_step, runner_state, None, rollout_steps)
    agent_state,env_state,env_params,last_obs, rng = runner_state
    
    # Calculate last value
    train_state = agent_state.network_state
    _,last_value = train_state.apply_fn(train_state.params, jnp.expand_dims(last_obs,(0,1)))
    last_val = last_value.squeeze()
    
    # Update step
    agent_state, loss_info, rng= update_step(traj_batch, agent_state, last_val, rng)
    
    ## Logging
    metric = traj_batch.info
    jax.debug.callback(wandb_ppo_logging, (metric,loss_info))
    
    ## Update runner state
    runner_state = (agent_state, env_state, env_params, last_obs, rng)
    return runner_state, metric


def experiment(rng,config:ExpConfig):
    ppo_config = config.ppo_config
    env_config = config.env_config
    
    # Create and initialize the environment.
    env,env_params = make_env(env_config,ppo_config.gamma) ## the gamma is needed for reward normalization wrapper
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng, env_params)
    
    # Initialize the agent 
    action_dim = None
    if env_config.continous_action:
        action_dim = env.action_space(env_params).shape[0]
    else:
        action_dim = env.action_space(env_params).n
    
    rng, _rng = jax.random.split(rng)
    agent_state = init_agent_state(ppo_config,action_dim,env.observation_space(env_params).shape,env_config.continous_action, _rng)
    
    runner_state = (agent_state, env_state, env_params, obs, rng)
    runner_state, metric = jax.lax.scan(partial(exp_step,env=env), runner_state, None, ppo_config.num_updates)
    return {"runner_state": runner_state, "metrics": metric}
                  
def main():
    # Reading config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',type=str,default='experiments_configs/test_configs/gymnax_config.yaml')
    args = parser.parse_args()
    config = ExpConfig.from_yaml(args.config_file)
    rng = jax.random.PRNGKey(config.exp_seed)
    wandb.init(config=config,project=config.wandb_project_name)
    wandb.define_metric("env_steps")
    wandb.define_metric("return", step_metric="env_steps")
    _ = jax.block_until_ready(experiment(rng,config))
    wandb.finish()
if __name__ == "__main__":
    main()  