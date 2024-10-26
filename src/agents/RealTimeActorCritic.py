import flax.linen as nn 
from typing import Optional, Tuple, Union, Any, Sequence, Dict
from flax.linen.initializers import constant, orthogonal
import jax.numpy as jnp
import numpy as np
import distrax
import functools
from src.nets.rtus.rtus import *
from src.nets.MLP import MLP
from src.configs.PPOConfig import PPOConfig
from src.nets.lru.online_lru import *

'''
A modified version of Actor Critic To work with Real Time Recurrent Networks
'''

class RealTimeActorCritic(nn.Module):
    action_dim: Sequence[int]
    config: PPOConfig
    cont: bool
    @nn.compact
    def __call__(self, hidden, x):
        '''
        hidden: (batch_size, d_hidden)
        x: (obs, dones)
        obs: (batch_size, obs_dim)
        done: (batch_size,1)
        '''
        if self.config.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
            
        obs, dones = x
        #### Shared Representation (MLP + rec_fn) ####
        shared_repr = MLP(self.config.d_shared_repr, act=self.config.activation, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs) 
        shared_repr = activation(shared_repr)  #shared_repr: (batch_size, d_hidden)
        
        rec_fn = self.get_rec_fn()
        
        hidden = self.check_done(hidden,dones,obs.shape[0])
        hidden, shared_repr = rec_fn(self.config.d_rec)(hidden, shared_repr) #hidden: (batch_size, d_hidden) , shared_repr: (batch_size, d_hidden)
        
        #### Actor head ####
        actor_mean = nn.Dense(self.config.d_actor_head, kernel_init=orthogonal(2), bias_init=constant(0.0))(shared_repr)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean) #actor_mean: (batch_size, action_dim)
        
        if self.cont:
            actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        else:
            pi = distrax.Categorical(logits=actor_mean)

        #### Critic head ####
        critic = nn.Dense(self.config.d_critic_head, kernel_init=orthogonal(2), bias_init=constant(0.0))(shared_repr)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic) #critic: (batch_size, 1)
        return hidden, pi, jnp.squeeze(critic, axis=-1)
    
    
    def initialize_state(self,batch_size):
        # initialize hidden states and gradients for the recurrent network
        d_input = self.config.d_shared_repr[-1]
        carry_init = self.get_rec_fn().initialize_state(batch_size,self.config.d_rec,d_input)
        return carry_init


    def check_done(self,carry,dones,batch_size):
        # Resets hidden states when done is True
        #d_hidden = carry[0][0].shape[-1]
        #batch_size = carry[0][0].shape[0]
        temp_carry = self.initialize_state(batch_size)
        new_carry = jax.tree_util.tree_map(lambda x,y: jnp.where(jnp.expand_dims(dones,(np.arange(1,len(x.shape)-1))),x,y),temp_carry,carry)
        return new_carry
    
    def get_rec_fn(self):
        if self.config.rec_fn == 'linear_rtu':
            rec_fn = RealTimeLinearRTUsLayer
        elif self.config.rec_fn == 'non_linear_rtu':
            rec_fn = RealTimeNonLinearRTUsLayer
        elif self.config.rec_fn == 'online_proj_lru':
            rec_fn = OnlineProjLRULayer
        else:
            raise NotImplementedError
        return rec_fn