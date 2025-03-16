import flax.linen as nn
import flax

# documents for custom_vjp
# jax) https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html
# flax) https://flax.readthedocs.io/en/v0.5.3/_autosummary/flax.linen.vjp.html


class RtrlRNNCellFwd(nn.Module):
    hidden_size: int
    output_dim: int

    @nn.compact
    def __call__(self, carry, x):
        prev_h, prev_sensitivity_matrix = carry
        rnn = nn.SimpleCell(features=self.hidden_size)

        # update hidden state
        curr_h, curr_out = rnn(prev_h, x)

        # update sensitivity matrix (TODO)
        curr_sensitivity_matrix = prev_sensitivity_matrix

        return (curr_h, curr_sensitivity_matrix), curr_out


class RtrlRNNModel(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, carry, x_t):
        def f(mdl, carry, x_t):
            return mdl(carry, x_t)

        def fwd(mdl, carry, x_t):
            f_out, vjp_func = nn.vjp(f, mdl, carry, x_t)
            # f_out = (rnn_out, sensitivity_matrix)
            # we need sensitivity_matrix for backward pass
            return f_out, (vjp_func, f_out[0][1])

        def bwd(residuals, y_t):
            vjp_func, sensitivity_matrix = residuals
            params_t, *inputs_t = vjp_func(y_t)
            params_t1 = flax.core.unfreeze(params_t)
            # fix params_t1
            return (params_t1, *inputs_t)

        vjp_fn = nn.custom_vjp(f, forward_fn=fwd, backward_fn=bwd)
        model_fn = RtrlRNNCellFwd(hidden_size=self.hidden_size)
        carry, hidden = vjp_fn(model_fn, carry, x_t)
        carry = carry[0]  # we don't need to forward sensitivity matrix to next module
        return carry, hidden
