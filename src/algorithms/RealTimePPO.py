"""
PPO with RTRL
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Any
from functools import partial
from src.configs.PPOConfig import PPOConfig
from flax.training.train_state import TrainState
import wandb
import numpy as np
from src.agents.RealTimeActorCritic import RealTimeActorCritic
import optax


PRNGKey = Any


class AgentState(NamedTuple):
    ppo_config: PPOConfig
    network_state: TrainState
    last_hstate: tuple  # hidden state of the rnn at time {t-1}


class Transition(NamedTuple):
    hidden_state: tuple  # hidden state of the rnn at time {t-1}
    obs: jnp.ndarray  # o_t
    done: jnp.ndarray  # done[t]
    action: jnp.ndarray  # a_t
    value: jnp.ndarray  # v(o_t)
    reward: jnp.ndarray  # r[t+1]
    log_prob: jnp.ndarray  # log_prob(a_t|o_t)
    info: jnp.ndarray


def init_agent_state(
    ppo_config: PPOConfig,
    action_dim: int,
    continous_action: bool,
    obs_shape: tuple,
    rng,
):
    # Create and initialize the network.
    actor_critic_net = RealTimeActorCritic(
        action_dim=action_dim, config=ppo_config, cont=continous_action
    )

    init_x = (jnp.zeros((1, *obs_shape)), jnp.zeros((1, 1)))
    init_hstate = actor_critic_net.initialize_state(1)
    network_params = actor_critic_net.init(rng, init_hstate, init_x)

    def params_sum(params):
        return sum(
            jax.tree_util.tree_leaves(jax.tree_map(lambda x: np.prod(x.shape), params))
        )

    if ppo_config.rec_fn == "linear_rtu":
        name = "RealTimeLinearRTUsLayer_0"
    elif ppo_config.rec_fn == "non_linear_rtu":
        name = "RealTimeNonLinearRTUsLayer_0"
    elif ppo_config.rec_fn == "online_proj_lru":
        name = "OnlineProjLRULayer_0"

    print("Total Number of params: %d" % params_sum(network_params))
    print("Recurrent Number of params: %d" % params_sum(network_params["params"][name]))
    wandb.summary["total_params"] = params_sum(network_params)
    wandb.summary["recurrent_params"] = params_sum(network_params["params"][name])
    if ppo_config.gradient_clipping:
        tx = optax.chain(
            optax.clip_by_global_norm(ppo_config.max_grad_norm),
            optax.adam(ppo_config.lr, eps=1e-5),
        )
    else:
        tx = optax.adam(ppo_config.lr, eps=1e-5)

    network_state = TrainState.create(
        apply_fn=actor_critic_net.apply,
        params=network_params,
        tx=tx,
    )

    return AgentState(
        ppo_config=ppo_config, network_state=network_state, last_hstate=init_hstate
    )


@jax.jit
def agent_step(
    agent_state: AgentState, obs_t: jnp.ndarray, done_t: jnp.ndarray, rng: PRNGKey
):
    rnn_in = (jnp.expand_dims(obs_t, 0), jnp.expand_dims(done_t, (0, 1)))
    last_hidden, pi, value = agent_state.network_state.apply_fn(
        agent_state.network_state.params, agent_state.last_hstate, rnn_in
    )
    action = pi.sample(seed=rng)
    log_prob = pi.log_prob(action)
    agent_state = agent_state._replace(last_hstate=last_hidden)
    return action, value, log_prob, agent_state


@jax.jit
def agent_update(
    agent_state: AgentState, traj_batch: list, last_obs, last_done, rng: PRNGKey
):
    # Calculate the advantages and value targets
    advantages, targets = _calculate_targets(
        agent_state, traj_batch, last_obs, last_done
    )
    # Learning
    update_state = (
        agent_state,
        traj_batch,
        advantages,
        targets,
        last_obs,
        last_done,
        rng,
    )
    update_state, loss_info = jax.lax.scan(
        _update_step, update_state, None, agent_state.ppo_config.epochs
    )

    agent_state = update_state[0]
    rng = update_state[-1]
    return agent_state, rng, loss_info


@jax.jit
def _update_step(update_state, unused):
    agent_state, traj_batch, advantages, targets, last_obs, last_done, rng = (
        update_state
    )
    batch = (traj_batch, advantages, targets)
    # Prepare minibatches
    minibatches_info, rng = _create_minibaches(agent_state, batch, rng)
    # TODO: if bptt or bptt_rtrl -> don't do the squeeze
    minibatches_info = jax.tree_util.tree_map(lambda x: x.squeeze(2), minibatches_info)

    # Loop through minibatches
    agent_state, (total_loss, grad_norm) = jax.lax.scan(
        _update_minbatch, agent_state, minibatches_info
    )
    update_state = (
        agent_state,
        traj_batch,
        advantages,
        targets,
        last_obs,
        last_done,
        rng,
    )

    refresh_hstates_and_maybe_targets = not agent_state.ppo_config.stale_gradient
    refresh_target_only = (
        not agent_state.ppo_config.stale_target
        and agent_state.ppo_config.stale_gradient
    )

    update_state = jax.lax.cond(
        refresh_target_only, _refresh_targets_only, lambda x: x, update_state
    )

    update_state = jax.lax.cond(
        refresh_hstates_and_maybe_targets,
        _refresh_hstates_and_maybe_targets,
        lambda x: x,
        update_state,
    )

    return update_state, (total_loss, grad_norm)


@jax.jit
def _refresh_targets_only(update_state):
    agent_state, traj_batch, advantages, targets, last_obs, last_done, rng = (
        update_state
    )

    # update trajectory values and action probabilities
    rnn_in = (traj_batch.obs, jnp.expand_dims(traj_batch.done, 1))
    hidden_states = jax.tree_util.tree_map(
        lambda x: x.squeeze(1), traj_batch.hidden_state
    )
    _, _, value = agent_state.network_state.apply_fn(
        agent_state.network_state.params, hidden_states, rnn_in
    )
    traj_batch = traj_batch._replace(value=value)

    # Calculate the advantages and value targets
    advantages, targets = _calculate_targets(
        agent_state, traj_batch, last_obs, last_done
    )
    update_state = (
        agent_state,
        traj_batch,
        advantages,
        targets,
        last_obs,
        last_done,
        rng,
    )
    return update_state


@jax.jit
def _refresh_hstates_and_maybe_targets(update_state):
    agent_state, traj_batch, advantages, targets, last_obs, last_done, rng = (
        update_state
    )

    def _get_hiddens(in_hidden, inputs):
        obs, done, _ = inputs
        hiddens, _, value = agent_state.network_state.apply_fn(
            agent_state.network_state.params, in_hidden, (obs, done)
        )
        return hiddens, (hiddens, value)

    # refresh the hiddens states
    in_hidden = jax.tree_map(lambda x: x[0, :], traj_batch.hidden_state)
    _, (hstates, value) = jax.lax.scan(
        _get_hiddens,
        in_hidden,
        (
            jnp.expand_dims(traj_batch.obs, 1),
            jnp.expand_dims(traj_batch.done, (1, 2)),
            traj_batch.action,
        ),
    )

    # refresh hidden states
    traj_batch = traj_batch._replace(hidden_state=hstates)

    # refresh the targets and advantages
    def _update_targets(ins):
        traj_batch, _, _ = ins
        traj_batch = traj_batch._replace(value=jnp.squeeze(value))
        # Calculate the advantages and value targets
        advantages, targets = _calculate_targets(
            agent_state, traj_batch, last_obs, last_done
        )
        return (traj_batch, advantages, targets)

    (traj_batch, advantages, targets) = jax.lax.cond(
        agent_state.ppo_config.stale_target,
        lambda x: x,
        _update_targets,
        (traj_batch, advantages, targets),
    )

    update_state = (
        agent_state,
        traj_batch,
        advantages,
        targets,
        last_obs,
        last_done,
        rng,
    )
    return update_state


@jax.jit
def _update_minbatch(agent_state, batch_info):
    config = agent_state.ppo_config
    train_state = agent_state.network_state
    grad_fn = jax.value_and_grad(_ppo_rnn_loss_fn, has_aux=True)
    total_loss, grads = grad_fn(
        train_state.params,
        train_state.apply_fn,
        batch_info,
        config.clip_eps,
        config.vf_coef,
        config.entropy_coef,
    )
    train_state = train_state.apply_gradients(grads=grads)
    agent_state = agent_state._replace(network_state=train_state)
    ## diagnostics
    # new_params = train_state.params
    grad_norm = tree_norm(grads)
    # cos_sim = cos_sim_pytrees(old_params,new_params)
    return agent_state, (total_loss, grad_norm)


@jax.jit
def _create_minibaches(agent_state, batch, rng):
    """Create minibatches from a batch of trajectories"""
    """ Logic:
    1. Divide the trajectories into number of sequences
    2. Shuffle the sequences
    3. Divide the sequences into minibatches
    The output will have the shape of (num_minibatches, minibatch_size, seq_len,_)
    """
    traj_batch, advantages, targets = batch
    config = agent_state.ppo_config
    hidden_state = jax.tree_util.tree_map(
        lambda y: jnp.squeeze(y, axis=1), traj_batch.hidden_state
    )
    traj_batch = traj_batch._replace(hidden_state=hidden_state)
    batch = (traj_batch, advantages, targets)
    number_sequences = config.rollout_steps // config.seq_len_in_minibatch
    minibatch_size = config.rollout_steps // (
        config.seq_len_in_minibatch * config.num_mini_batch
    )
    # reshape the batch to (number_sequences, seq_len, _)
    batch = jax.tree_util.tree_map(
        lambda x: x.reshape(
            (
                number_sequences,
                config.seq_len_in_minibatch,
            )
            + x.shape[1:]
        ),
        batch,
    )
    # shuffle the sequences
    rng, _rng = jax.random.split(rng)
    permutation = jax.random.permutation(_rng, number_sequences)
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: jnp.take(x, permutation, axis=0), batch
    )
    # reshape the shuffled batch to (num_minibatches, minibatch_size, seq_len,_)
    batch = jax.tree_util.tree_map(
        lambda x: x.reshape(
            (
                config.num_mini_batch,
                minibatch_size,
            )
            + x.shape[1:]
        ),
        shuffled_batch,
    )
    return batch, rng


def wandb_ppo_logging(info_and_loss):
    info, loss_info = info_and_loss
    loss_info, grad_norm = loss_info
    return_values = info["returned_episode_returns"]
    timesteps = info["timestep"]
    wandb.log(
        {"env_steps": np.mean(timesteps), "undiscounted_return": np.mean(return_values)}
    )
    wandb.log(
        {
            "total_loss": np.mean(np.mean(loss_info[0])),
            "value_loss": np.mean(np.mean(loss_info[1][0])),
            "policy_loss": np.mean(np.mean(loss_info[1][1])),
            "entropy": np.mean(np.mean(loss_info[1][2])),
            "k1": np.mean(np.mean(loss_info[1][3])),
            "k2": np.mean(np.mean(loss_info[1][4])),
            "k3": np.mean(np.mean(loss_info[1][5])),
            "ratio": np.mean(np.mean(loss_info[1][6])),
            "explained_variance": np.mean(np.mean(loss_info[1][7])),
            "grad_norm": np.mean(grad_norm),
        }
    )
    #'cos_sim': np.mean(cos_sim)})


@partial(jax.jit, static_argnums=(1,))
def _ppo_rnn_loss_fn(params, agent_fn, minibatch, clip_eps, vf_coef, ent_coef):
    traj_batch, gae, targets = minibatch

    # forward pass
    rnn_in = (traj_batch.obs, jnp.expand_dims(traj_batch.done, 1))
    _, pi, value = agent_fn(params, traj_batch.hidden_state, rnn_in)
    log_prob = pi.log_prob(traj_batch.action)

    # Calculate clipped value loss
    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
        -clip_eps, clip_eps
    )
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    ## KL diagnostics
    # check http://joschu.net/blog/kl-approx.html for reference
    # r = p(x) / q(x) => for samples x coming from q
    # k1 = - log r
    # k2 =  (log r)^2 /2
    # k3 =  (r - 1) - log r

    log_r = log_prob - traj_batch.log_prob
    k1 = (-log_r).mean()
    k2 = ((log_r**2) / 2).mean()
    k3 = ((jnp.exp(log_r) - 1) - log_r).mean()

    # calculate policy loss
    ratio = jnp.exp(log_prob - traj_batch.log_prob)
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
    entropy = pi.entropy().mean()
    total_loss = loss_actor + vf_coef * value_loss - ent_coef * entropy
    # explained variance
    explained_variance = 1 - jnp.var(targets - value) / jnp.var(targets)
    return total_loss, (
        value_loss,
        loss_actor,
        entropy,
        k1,
        k2,
        k3,
        ratio.mean(),
        explained_variance.mean(),
    )


@jax.jit
def _calculate_targets(agent_state, traj_batch, last_obs, last_done):
    # Calculate the advantages and value targets

    rnn_in = (jnp.expand_dims(last_obs, 0), jnp.expand_dims(last_done, (0, 1)))
    _, _, last_value = agent_state.network_state.apply_fn(
        agent_state.network_state.params, agent_state.last_hstate, rnn_in
    )
    last_val = last_value.squeeze()

    ## Calculate targets and advantages
    advantages, targets = _calculate_gae(
        traj_batch,
        last_val,
        last_done,
        agent_state.ppo_config.gamma,
        agent_state.ppo_config.gae_lambda,
    )
    return advantages, targets


@jax.jit
def _calculate_gae(traj_batch, last_val, last_done, gamma, gae_lambda):
    """Calculate advantages and value targets
    GAE_t = delta_t + gamma * lambda * (1 - done_{t+1}) * GAE_{t+1}
    GAE_{traj_len+1} = 0
    delta_t = reward_t + gamma * value_{t+1} * (1 - done_{t+1}) - value_t
    Args:
        traj_batch (list): A list of Transitions with shape (num_steps, _)
        last_val (float): Value of the last observation. This is the observation that follows the last observation in traj_batch.
        last_done (bool): Done flag of the last observation. This is the observation that follows the last observation in traj_batch.
        gamma (float): Discount factor
        gae_lambda (float): GAE lambda
    Returns:
        advantages (jnp.ndarray): Advantages with shape (num_steps, _)
        targets (jnp.ndarray): Value targets with shape (num_steps, _)
    """

    def _get_advantages(carry, transition):
        gae, next_value, next_done = carry
        done, value, reward = (
            transition.done,
            transition.value,
            transition.reward,
        )
        delta = reward + gamma * next_value * (1 - next_done) - value
        gae = delta + gamma * gae_lambda * (1 - next_done) * gae
        return (gae, value, done), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val, last_done),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj_batch.value


@jax.jit
def tree_norm(x):
    return jnp.sqrt(
        jax.tree_util.tree_reduce(jnp.add, jax.tree_map(lambda x: jnp.sum(x**2), x))
    )


@jax.jit
def cos_sim_pytrees(old_params, new_params):
    mul = jax.tree_map(lambda x, y: x * y, old_params, new_params)
    dot_prod = jax.tree_util.tree_reduce(
        jnp.add, jax.tree_map(lambda x: jnp.sum(x), mul)
    )
    old_norm = tree_norm(old_params)
    new_norm = tree_norm(new_params)
    return dot_prod / (old_norm * new_norm)
