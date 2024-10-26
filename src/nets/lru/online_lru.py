import sys
sys.path.append('./')
sys.path.append('../')
from flax import linen as nn
import jax 
import jax.numpy as jnp 
import flax 
from typing import Callable, Any, Tuple, Iterable,Optional
from src.nets.lru.params_init import *
from functools import partial

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  
Array = Any


def get_lambda(nu_log, theta_log):
    Lambda = jnp.exp(-jnp.exp(nu_log) + 1j* jnp.exp(theta_log))
    return Lambda
        
class FwdOnlineLRU(nn.Module):
    d_hidden: int
    r_max: jnp.float32 = 1.0
    r_min: jnp.float32 = 0.0    
    max_phase : jnp.float32 = 6.28
    '''
    grad memory: dh_{t-1}/d lambda, dh_{t-1}/d gamma #1,  
                 dh_c_{t-1}/d B #2
    '''
    def setup(self):
        self.nu_log = self.param("nu_log", nu_log_init, (1,self.d_hidden), self.r_max, self.r_min)
        self.theta_log = self.param("theta_log", theta_log_init, (1,self.d_hidden), self.max_phase)
        self.gamma_log = self.param("gamma_log", gamma_log_init, (1,self.d_hidden), self.nu_log, self.theta_log)
    
    @nn.compact
    def __call__(self, carry,inputs):
        h_tminus1,grad_memory = carry 
        input_dim = inputs.shape[-1]
        hidden_dim = h_tminus1.shape[-1]

        B_real = self.param(
            "B_real",
            partial(matrix_init, normalization=jnp.sqrt(2 * input_dim)),
            (hidden_dim, input_dim),
        )

        B_img = self.param(
            "B_img",
            partial(matrix_init, normalization=jnp.sqrt(2 * input_dim)),
            (hidden_dim, input_dim),
        )
        
        Lambda = get_lambda(self.nu_log,self.theta_log)
        B = (B_real + 1j * B_img)
        
        B_norm = B * jnp.exp(jnp.expand_dims(self.gamma_log,axis=-1))
        
        h_t = (Lambda * h_tminus1) + (inputs @ B_norm.squeeze().transpose())
        
        new_grad_lambda = Lambda * grad_memory[0] + h_tminus1
        

        new_grad_gamma = Lambda * grad_memory[1] + (inputs @ jnp.swapaxes(B,-1,-2)).squeeze()
        
        new_grad_B = (jnp.expand_dims(Lambda, axis=-1)) * grad_memory[2] + jax.vmap(lambda u: jnp.outer(self.gamma_log, u))(
                    inputs).astype(jnp.complex64)

        new_grad_memory = (
            new_grad_lambda,
            new_grad_gamma,
            new_grad_B,
        )
        new_carry = (h_t, new_grad_memory)
        return new_carry, new_carry

    def to_lambda(self, x):
        return get_lambda(self.nu_log, self.theta_log)
    

class OnlineLRUCell(nn.Module):
    d_hidden: int
    @nn.compact
    def __call__(self,carry,x_t):
        def f(mdl,carry,x_t):
            return mdl(carry,x_t)
        
        def fwd(mdl,carry,x_t):
            f_out, vjp_func = nn.vjp(f, mdl, carry, x_t)
            _, vjp_to_lambda = nn.vjp(lambda m, x: m.to_lambda(x), mdl, x_t)
            return f_out, (vjp_func, f_out[1][1], vjp_to_lambda,mdl.gamma_log)  # output, residual
        
        def bwd(residuals, y_t): 
            #y_t =(partial{output}/partial{h_{t}},ignore the rest
            #grad_memory = \partial{h_{t-1}} \partial{lambda},
                           # \partial{h_{t-1},c1} \partial{gamma},\partial{h_{t-1}} \partial{B}
            vjp_func, new_grad_memory, vjp_to_lambda,gamma_log = residuals
            params_t,*inputs_t = vjp_func(y_t)
            d_output_d_h = y_t[1][0]
            
            d_output_d_lambda = d_output_d_h * new_grad_memory[0]
            d_params_rec = vjp_to_lambda(jnp.expand_dims(jnp.sum(d_output_d_lambda,0),0))[0]
            correct_nu_log, correct_theta_log = (
                d_params_rec["params"]["nu_log"],
                d_params_rec["params"]["theta_log"],
            )
             
            correct_gamma_log = (d_output_d_h * new_grad_memory[1]).real * jnp.exp(gamma_log)
            grad_B = jnp.sum((jnp.expand_dims(d_output_d_h,-1) * new_grad_memory[2]),0)
            params_t1 = flax.core.unfreeze(params_t)
            params_t1['params']['nu_log'] = correct_nu_log 
            params_t1['params']['theta_log'] = correct_theta_log
            params_t1['params']['gamma_log'] = (jnp.expand_dims(jnp.sum(correct_gamma_log,0),0)).real
            params_t1['params']['B_real'] = grad_B.real
            params_t1['params']['B_img'] = -grad_B.imag
            return (params_t1, *inputs_t)
        online_lru_cell_grad = nn.custom_vjp(f, forward_fn=fwd, backward_fn=bwd)
        model_fn = FwdOnlineLRU(self.d_hidden)
        (h_t,new_grad_memory),(h_t,new_grad_memory) = online_lru_cell_grad(model_fn, carry,x_t)
        return (h_t,new_grad_memory),(h_t) # carry, output
    
'''
OnlineLRU layer with linear projection afterwards
'''
class OnlineProjLRULayer(nn.Module):
    d_hidden: int
    @nn.compact
    def __call__(self,carry,x_t):
        h_tminus1,_ = carry 
        hidden_dim = h_tminus1.shape[-1]

        C_real = self.param(
            "C_real",
            partial(matrix_init, normalization=jnp.sqrt(hidden_dim)),
            (self.d_hidden, hidden_dim),
        )

        C_img = self.param(
            "C_img",
            partial(matrix_init, normalization=jnp.sqrt(hidden_dim)),
            (self.d_hidden, hidden_dim),
        )

        
        online_lru = OnlineLRUCell(self.d_hidden)
        carry,h_t  = online_lru(carry,x_t)
        C = (C_real + 1j * C_img)
        y_t = (h_t @ C.transpose()).real
        
        return carry,y_t # carry, output
    
    @staticmethod
    def initialize_state(batch_size,d_hidden, d_input):
        hidden_init = jnp.zeros((batch_size,d_hidden), dtype=jnp.complex64)
        memory_grad_init = (
            jnp.zeros((batch_size,d_hidden), dtype=jnp.complex64),
            jnp.zeros((batch_size,d_hidden), dtype=jnp.complex64),
            jnp.zeros((batch_size,d_hidden,d_input), dtype=jnp.complex64),
        )
        return (hidden_init, memory_grad_init)

if __name__=="__main__":
    input_dim = 2
    d_hidden = 5
    seq_len = 2
    batch_size = 2
    

    h_init,grad_momory_init = OnlineProjLRULayer.initialize_state(batch_size,d_hidden,input_dim) 
    inputs = jnp.ones((batch_size,input_dim),dtype=jnp.float32)
    model = OnlineProjLRULayer(d_hidden=d_hidden)
    params = model.init(jax.random.PRNGKey(0),(h_init,grad_momory_init),inputs)
    print(params)
    
    test_x = jax.random.normal(jax.random.PRNGKey(0),(seq_len,batch_size,input_dim)) 
    #print(test_x)
    def apply_rtrl(rt_rtu_params,test_x,rt_hidden_init,mem_grad_init):
        rt_carry = (rt_hidden_init,mem_grad_init)
        hs_c1 = []
        for i in range(seq_len):
            rt_carry,out = model.apply(rt_rtu_params,rt_carry,test_x[i,:,:])
            hs_c1.append(out)
        error = (1 - jnp.mean(jnp.stack(hs_c1)))**2
        return error
    rtrl_out = apply_rtrl(params,test_x,h_init,grad_momory_init)
    grad_rtrl = jax.grad(apply_rtrl)(params,test_x,h_init,grad_momory_init)
    print(rtrl_out)
    print(grad_rtrl)
    