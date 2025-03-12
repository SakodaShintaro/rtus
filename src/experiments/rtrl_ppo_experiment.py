import sys

sys.path.append("./")
sys.path.append("../")

import jax
from flax import struct
from typing import Any, Callable

from src.envs.wrappers.purerl import *
from src.configs.ExpConfig import ExpConfig
import wandb
import argparse
from timeit import default_timer as timer
from src.algorithms.RealTimePPO import *
from src.envs.make_envs import *

# jax.config.update("jax_disable_jit", True)


class GymnaxEnvState(struct.PyTreeNode):
    env_step: Callable = struct.field(pytree_node=False)
    env_params: Any = struct.field(pytree_node=True)
    env_state: Any = struct.field(pytree_node=True)

    @classmethod
    def create(cls, *, env_step, env_params, env_state, **kwargs):
        """Creates a new instance"""
        return cls(
            env_step=env_step,
            env_params=env_params,
            env_state=env_state,
            **kwargs,
        )


@jax.jit
def env_step(runner_state, unused):
    agent_state, gymnax_state, last_obs, last_done, rng = runner_state
    # agent step
    rng, _rng = jax.random.split(rng)
    in_hstate = agent_state.last_hstate
    action, value, log_prob, agent_state = agent_step(
        agent_state, last_obs, last_done, _rng
    )
    # env step
    rng, rng_step = jax.random.split(rng)
    obs, env_state, reward, done, info = gymnax_state.env_step(
        rng_step, gymnax_state.env_state, action.squeeze(), gymnax_state.env_params
    )

    # store transition
    transition = Transition(
        in_hstate,
        last_obs,
        last_done,
        action.squeeze(),
        value.squeeze(),
        reward,
        log_prob.squeeze(),
        info,
    )
    gymnax_state = GymnaxEnvState.create(
        env_step=gymnax_state.env_step,
        env_params=gymnax_state.env_params,
        env_state=env_state,
    )
    runner_state = (agent_state, gymnax_state, obs, done, rng)
    return runner_state, transition


@jax.jit
def experiment_step(runner_state, unused):
    # Rollout
    rollout_steps = runner_state[0].ppo_config.rollout_steps
    runner_state, traj_batch = jax.lax.scan(env_step, runner_state, None, rollout_steps)
    agent_state, gymnax_state, last_obs, last_done, rng = runner_state

    # Update
    agent_state, rng, loss_info = agent_update(
        agent_state, traj_batch, last_obs, last_done, rng
    )

    # Logging
    metric = traj_batch.info
    jax.debug.callback(wandb_ppo_logging, (metric, loss_info))

    # Update runner state
    runner_state = (agent_state, gymnax_state, last_obs, last_done, rng)
    return runner_state, metric


def experiment(rng, config: ExpConfig):
    ppo_config = config.ppo_config
    env_config = config.env_config

    # Create and initialize the environment.
    env, env_params = make_env(
        env_config, ppo_config.gamma
    )  ## the gamma is needed for reward normalization wrapper
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng, env_params)
    gymnax_state = GymnaxEnvState.create(
        env_step=env.step, env_params=env_params, env_state=env_state
    )

    # Create and initialize the agent.
    action_dim = None
    if env_config.continous_action:
        action_dim = env.action_space(env_params).shape[0]
    else:
        action_dim = env.action_space(env_params).n

    obs_shape = env.observation_space(env_params).shape

    rng, _rng = jax.random.split(rng)
    agent_state = init_agent_state(
        ppo_config, action_dim, env_config.continous_action, obs_shape, _rng
    )

    # start the experiment
    runner_state = (agent_state, gymnax_state, obs, False, rng)
    runner_state, metric = jax.lax.scan(
        experiment_step, runner_state, None, ppo_config.num_updates
    )
    return {"runner_state": runner_state, "metrics": metric}


def main():
    # Reading config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, default="configs/pobrax_config_rtu_rtrl.yaml"
    )
    args = parser.parse_args()
    config = ExpConfig.from_yaml(args.config_file)
    rng = jax.random.PRNGKey(config.exp_seed)
    wandb.init(config=config, project=config.wandb_project_name)
    wandb.define_metric("env_steps")
    wandb.define_metric("undiscounted_return", step_metric="env_steps")
    start = timer()
    _ = jax.block_until_ready(experiment(rng, config))
    end = timer()
    elapsed_time = end - start
    wandb.summary["elapsed_time"] = elapsed_time
    wandb.finish()


if __name__ == "__main__":
    main()
