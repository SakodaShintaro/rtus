from flax import struct
from src.configs.PPOConfig import PPOConfig
from typing import NamedTuple, Any, Callable
import jax.numpy as jnp
from flax.training.train_state import TrainState
from src.agents.ActorCritic import ActorCritic
import jax
import optax 
import numpy as np
from functools import partial
import wandb 
PRNGKey = Any


class AgentState(NamedTuple):
    ppo_config: PPOConfig
    network_state: TrainState

class Transition(NamedTuple):
    obs: jnp.ndarray    # o_t
    action: jnp.ndarray # a_t
    reward: jnp.ndarray # r[t+1]
    done: jnp.ndarray   # done[t+1]
    value: jnp.ndarray  # v(o_t)
    log_prob: jnp.ndarray # log_prob Pi(a_t|o_t)
    info: jnp.ndarray



def init_agent_state(configs: PPOConfig, action_dim:int,obs_shape:tuple,continous_action:bool, rng:PRNGKey):
    # create network
    network = ActorCritic(
            action_dim = action_dim, 
            activation=configs.activation,
            actor_hiddens=configs.d_actor_head,
            critic_hiddens=configs.d_critic_head,
            cont=continous_action)

    
    #input shape = (batch_size, obs_dim)
    init_x = jnp.zeros((1,*obs_shape))
    network_params = network.init(rng, init_x)
    def params_sum(params):
            return sum(jax.tree_util.tree_leaves(jax.tree_map(lambda x: np.prod(x.shape),params)))
    print("Total Number of params: %d"%params_sum(network_params))
     
    if configs.gradient_clipping:
            tx = optax.chain(
                optax.clip_by_global_norm(configs.max_grad_norm),
                optax.adam(configs.lr, eps=1e-5),
            )
    else:
        tx = optax.adam(configs.lr, eps=1e-5)
    network_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )
    
    return AgentState(ppo_config=configs, network_state=network_state)
        
@jax.jit
def agent_step(agent_state: AgentState, obs: jnp.ndarray, rng: PRNGKey) :
    '''
    obs.shape = (obs_dim,)
    '''
    pi, value = agent_state.network_state.apply_fn(agent_state.network_state.params,jnp.expand_dims(obs,(0,1)))
    action = pi.sample(seed=rng)
    log_prob = pi.log_prob(action)  
    return action, value, log_prob    

@jax.jit
def update_step(traj_batch, agent_state, last_val, rng):
    update_data = (traj_batch, agent_state, last_val, rng)
                   
    agent_state, loss_info,rng = jax.lax.cond(agent_state.ppo_config.stale_target,
                 _update_step_stale_target,
                 _update_step_fresh_target,
                update_data)
    return agent_state, loss_info,rng


@partial(jax.jit, static_argnums=(1,))
def _ppo_loss_fn(params,agent_fn, traj_batch, gae, targets,clip_eps,vf_coef,ent_coef):
    """PPO loss function with stale targets
    Logic:
    1. Re-run the network through the batch with the current policy to get:
        a.The value of the old observations according to the current value function
        b.The log probability of the old actions according to the current policy
    2. Estimate value loss/actor loss/entropy 
    Args:
        params (dict): Network parameters
        agent_fn (Callable): Network function
        traj_batch (list): List of Transitions with shape (minibatch_size,seq_len, _)
        gae (list): List of advantages with shape (minibatch_size,seq_len, _)
        targets (list): List of value targets with shape (minibatch_size,seq_len, _)
        clip_eps (float): PPO clip coefficient
        vf_coef (float): Value loss coefficient
        ent_coef (float): Entropy coefficient
    """
    # Re-run the network through the batch
    pi, value = agent_fn(params, traj_batch.obs)
    log_prob = pi.log_prob(traj_batch.action)
    # Calculate value loss
    value_pred_clipped = traj_batch.value + (
        value - traj_batch.value
    ).clip(-clip_eps, clip_eps)
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = (
        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    )
    # Calculate actor loss
    log_ratio = log_prob - traj_batch.log_prob
    ratio = jnp.exp(log_ratio)
    approx_kl = ((ratio - 1) - log_ratio).mean()
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = (
        jnp.clip(
            ratio,
            1.0 - clip_eps,
            1.0 + clip_eps,
        )
        * gae
    )
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()
    # Calculate entropy
    entropy = pi.entropy().mean()
    total_loss = (
        loss_actor
        + vf_coef * value_loss
        - ent_coef * entropy
    )
    return total_loss, (value_loss, loss_actor, entropy,(-log_ratio).mean(),approx_kl)
    
@jax.jit
def _update_minibatch(carry_in, batch_info):
    train_state,config = carry_in
    traj_batch, advantages, targets = batch_info
    grad_fn = jax.value_and_grad(_ppo_loss_fn, has_aux=True)
    total_loss, grads = grad_fn(
        train_state.params,train_state.apply_fn, traj_batch, advantages, targets,
        config.clip_eps,config.vf_coef,config.entropy_coef
    )
    train_state = train_state.apply_gradients(grads=grads)
    return (train_state,config), total_loss

@jax.jit
def _calculate_gae(traj_batch, last_val,gamma,gae_lambda):
    """Calculate advantages and value targets
    GAE_t = delta_t + gamma * lambda * (1 - done_{t+1}) * GAE_{t+1}
    GAE_{traj_len+1} = 0
    delta_t = reward_t + gamma * value_{t+1} * (1 - done_{t+1}) - value_t
    Args:
        traj_batch (list): A list of Transitions with shape (num_steps, _)
        last_val (float): Value of the last observation. This is the observation that follows the last observation in traj_batch.
        gamma (float): Discount factor
        gae_lambda (float): GAE lambda
    Returns:
        advantages (jnp.ndarray): Advantages with shape (num_steps, _)
        targets (jnp.ndarray): Value targets with shape (num_steps, _)
    """
    def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = (
                transition.done,
                transition.value,
                transition.reward,
            )
            delta = reward + gamma * next_value * (1 - done) - value
            gae = (
                delta
                + gamma * gae_lambda * (1 - done) * gae
            )
            return (gae, value), gae
    _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
    return advantages, advantages + traj_batch.value  
  
@jax.jit
def _create_minibaches(config,batch, rng):
    """Create minibatches from a batch of trajectories """
    """ Logic:
    1. Divide the trajectories into number of sequences
    2. Shuffle the sequences 
    3. Divide the sequences into minibatches
    The output will have the shape of (num_minibatches, minibatch_size, seq_len,_)
    """
    number_sequences = config.rollout_steps // config.seq_len_in_minibatch 
    minibatch_size = config.rollout_steps//(config.seq_len_in_minibatch*config.num_mini_batch)
    # reshape the batch to (number_sequences, seq_len, _)
    batch = jax.tree_util.tree_map(
        lambda x: x.reshape((number_sequences,config.seq_len_in_minibatch,)+x.shape[1:]), batch)
    # shuffle the sequences
    rng, _rng = jax.random.split(rng)
    permutation = jax.random.permutation(_rng, number_sequences)
    shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch)
    # reshape the shuffled batch to (num_minibatches, minibatch_size, seq_len,_)
    batch = jax.tree_util.tree_map(
        lambda x: x.reshape((config.num_mini_batch,minibatch_size,)+x.shape[1:]), shuffled_batch)
    return batch,rng

@jax.jit
def _batch_update(update_state, unused):
    train_state, traj_batch, advantages, targets, rng,config = update_state
    batch = (traj_batch, advantages, targets)
    
    minibatches_info,rng = _create_minibaches(config,batch, rng)
    # Loop through minibatches 
    carry_in = (train_state,config)
    carry_out, total_loss = jax.lax.scan(_update_minibatch, carry_in, minibatches_info)
    train_state = carry_out[0]
    update_state = (train_state, traj_batch, advantages, targets, rng,config)
    return update_state, total_loss

@jax.jit
def _update_epoch_fresh_targets(update_state, unused):
    train_state, traj_batch, last_val, rng,config = update_state
    # calculate advantages and targets
    advantages, targets = _calculate_gae(traj_batch, last_val,config.gamma,config.gae_lambda)
    update_state = train_state, traj_batch, advantages, targets, rng,config 
    update_state, total_loss = _batch_update(update_state, unused) 
    new_update_state = (update_state[0],traj_batch, last_val, rng,config)
    return new_update_state, total_loss


@jax.jit
def _update_step_stale_target(update_data):
    traj_batch, agent_state, last_val, rng = update_data
    # calculate advantages and targets
    config = agent_state.ppo_config
    advantages, targets = _calculate_gae(traj_batch, last_val,config.gamma,config.gae_lambda)
    # Learning 
    update_state = (agent_state.network_state,traj_batch, advantages, targets, rng,config)
    update_state, loss_info = jax.lax.scan(_batch_update, update_state, None, config.epochs)
    agent_state = AgentState(ppo_config=config, network_state=update_state[0])
    return agent_state, loss_info,update_state[-2]

@jax.jit
def _update_step_fresh_target(update_data):
    traj_batch, agent_state, last_val, rng = update_data
    update_state = agent_state.network_state, traj_batch, last_val, rng, agent_state.ppo_config
    config = agent_state.ppo_config 
    update_state, loss_info = jax.lax.scan(_update_epoch_fresh_targets, update_state, None, config.epochs)
    agent_state = AgentState(ppo_config=config, network_state=update_state[0])
    return agent_state, loss_info,update_state[-2]


def wandb_ppo_logging(info_and_loss):
    info, loss_info = info_and_loss
    return_values = info["returned_episode_returns"]
    timesteps = info["timestep"]
    wandb.log({'env_steps': np.mean(timesteps), 'undiscounted_return': np.mean(return_values)})
    wandb.log({'total_loss': np.mean(np.mean(loss_info[0])),
               'value_loss': np.mean(np.mean(loss_info[1][0])),
               'policy_loss': np.mean(np.mean(loss_info[1][1])),
               'entropy': np.mean(np.mean(loss_info[1][2])),
               'minus_log_ratio': np.mean(np.mean(loss_info[1][3])),
                'approx_kl': np.mean(np.mean(loss_info[1][4]))})
               