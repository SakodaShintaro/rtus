from collections.abc import Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

"""
Actor Critic Network with MLP as function approximator
Actor network:  (64, tanh, 64, tanh, Linear) + Standard deviation variable
Critic network: (64, tanh, 64, tanh, Linear)
"""


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    actor_hiddens: int = 64
    critic_hiddens: int = 64
    cont: bool = False

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        embedding = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        embedding = activation(embedding)
        embedding = nn.Dense(
            self.actor_hiddens, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = activation(embedding)
        actor_mean = nn.Dense(64, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        if self.cont:
            actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        else:
            pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            embedding
        )
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return pi, jnp.squeeze(critic, axis=-1)
