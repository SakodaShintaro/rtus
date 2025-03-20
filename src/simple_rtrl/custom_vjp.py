import flax.linen
import jax
import jax.numpy as jnp
from flax.training import train_state
import flax.linen as nn
import flax
import optax
import numpy as np
from functools import partial
from flax.linen.linear import Dense, default_kernel_init
from flax.typing import Dtype, Initializer
from flax.linen import initializers

# documents for custom_vjp
# jax) https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html
# flax) https://flax.readthedocs.io/en/v0.5.3/_autosummary/flax.linen.vjp.html

# 2つのクラスを実装する
# (1) Feedforwardだけを定義したRtrlRNNCellFwd
# (2) custom_vjpを定義したRtrlCell


def print_dict_tree(d, indent=0):
    for key, value in d.items():
        print("  " * indent + str(key))
        if isinstance(value, dict):
            print_dict_tree(value, indent + 1)


def print_tuple_tree(t, indent=0):
    for i, value in enumerate(t):
        print("  " * indent + str(i))
        if isinstance(value, tuple):
            print_tuple_tree(value, indent + 1)
        else:
            print("  " * (indent + 1) + str(value.shape))


class RtrlRNNCellFwd(nn.Module):
    hidden_size: int
    kernel_init: Initializer = default_kernel_init
    recurrent_kernel_init: Initializer = initializers.orthogonal()
    bias_init: Initializer = initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    carry_init: Initializer = initializers.zeros_init()
    residual: bool = False

    @staticmethod
    @jax.jit
    def dtanh(x):
        return 1 - jnp.tanh(x) ** 2

    @nn.compact
    def __call__(self, carry, x):
        prev_s, prev_sensitivity_matrices = carry
        prev_h = flax.linen.activation.tanh(prev_s)

        dense_h = partial(
            Dense,
            features=hidden_size,
            use_bias=False,
            param_dtype=self.param_dtype,
            kernel_init=self.recurrent_kernel_init,
        )
        dense_i = partial(
            Dense,
            features=hidden_size,
            use_bias=True,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

        # update hidden state
        curr_s = dense_i(name="i")(x) + dense_h(name="h")(prev_h)
        curr_out = flax.linen.activation.tanh(curr_s)

        # update sensitivity matrices
        R = self.variables["params"]["h"]["kernel"]
        (S_R, S_W, S_B) = prev_sensitivity_matrices
        d_s = self.dtanh(prev_s)
        eye = jnp.eye(hidden_size)
        S_W = jnp.einsum("bi,jk->bkij", x, eye) + jnp.einsum(
            "nk,bn,bnij->bkij", R, d_s, S_W
        )
        S_B = eye[None, :, :] + jnp.einsum("nk,bn,bnj->bkj", R, d_s, S_B)
        S_R = jnp.einsum("bi,jk->bkij", prev_h, eye) + jnp.einsum(
            "nk,bn,bnij->bkij", R, d_s, S_R
        )
        S_W = jax.lax.stop_gradient(S_W)
        S_B = jax.lax.stop_gradient(S_B)
        S_R = jax.lax.stop_gradient(S_R)
        curr_sensitivity_matrices = (S_R, S_W, S_B)

        return (curr_s, curr_sensitivity_matrices), curr_out


class RtrlCell(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, carry, x_t):
        def f(mdl, carry, x_t):
            return mdl(carry, x_t)

        def fwd(mdl, carry, x_t):
            f_out, vjp_func = nn.vjp(f, mdl, carry, x_t)
            # f_out = (curr_h, curr_sensitivity_matrices), curr_out
            # we need sensitivity_matrices for backward pass
            return f_out, (vjp_func, f_out[0][1])

        def bwd(residuals, y_t):
            vjp_func, sensitivity_matrices = residuals
            params_t, *inputs_t = vjp_func(y_t)
            params_t1 = flax.core.unfreeze(params_t)
            (S_R, S_W, S_B) = sensitivity_matrices
            y_t = y_t[0][0]
            curr_grad_W = jnp.einsum("bh,bhij->bij", y_t, S_W)
            curr_grad_B = jnp.einsum("bh,bhj->bj", y_t, S_B)
            curr_grad_R = jnp.einsum("bh,bhij->bij", y_t, S_R)
            params_t1["params"]["i"]["kernel"] = jnp.sum(curr_grad_W, 0)
            params_t1["params"]["i"]["bias"] = jnp.sum(curr_grad_B, 0)
            params_t1["params"]["h"]["kernel"] = jnp.sum(curr_grad_R, 0)
            return (params_t1, *inputs_t)

        vjp_fn = nn.custom_vjp(f, forward_fn=fwd, backward_fn=bwd)
        model_fn = RtrlRNNCellFwd(hidden_size=self.hidden_size)
        carry, hidden = vjp_fn(model_fn, carry, x_t)
        return carry, hidden

    @staticmethod
    def initialize_state(batch_size, d_rec, d_input):
        hidden_init = jnp.zeros((batch_size, d_rec))
        S_R = jnp.zeros((batch_size, d_rec, d_rec, d_rec))
        S_W = jnp.zeros((batch_size, d_rec, d_rec, d_input))
        S_B = jnp.zeros((batch_size, d_rec, d_input))
        memory_grad_init = (S_R, S_W, S_B)
        return (hidden_init, memory_grad_init)


def bptt_loss_fn(params, model, x, y):
    # x shape: [seq_len, batch_size, input_dim]
    batch_size = x.shape[1]

    # 最初のキャリー状態を初期化
    hidden_size = params["params"]["h"]["kernel"].shape[0]
    initial_carry = jnp.zeros((batch_size, hidden_size))

    # scanでシーケンス全体に対して処理を適用
    step_fn = partial(model, params)
    _, y_pred = jax.lax.scan(step_fn, initial_carry, x)

    # MSE損失を計算
    loss = jnp.mean((y_pred - y) ** 2)
    return loss


@jax.jit
def bptt_grads(state, batch_x, batch_y):
    loss, grads = jax.value_and_grad(bptt_loss_fn)(
        state.params, state.apply_fn, batch_x, batch_y
    )
    return loss, grads


def rtrl_grads(state, batch_x, batch_y):
    params = state.params
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
    n_params = flat_params.shape[0]
    grads_flat = jnp.zeros(n_params)
    seq_len = batch_x.shape[0]

    # 最初のキャリー状態を初期化
    hidden_size = params["params"]["RtrlRNNCellFwd_0"]["h"]["kernel"].shape[0]
    carry = RtrlCell.initialize_state(batch_size, hidden_size, hidden_size)

    loss = 0.0

    def step_loss_fn(params, carry, x_t, y_t_ref):
        carry, hidden = state.apply_fn(params, carry, x_t)
        curr_loss = jnp.mean((hidden - y_t_ref) ** 2)
        return curr_loss, (carry, hidden)

    for t in range(seq_len):
        print(f"{t=}")
        x_t = batch_x[t]
        y_t_ref = batch_y[t]
        (curr_loss, (carry, hidden)), grads = jax.value_and_grad(
            step_loss_fn, has_aux=True
        )(params, carry, x_t, y_t_ref)
        curr_flat_grads, _ = jax.flatten_util.ravel_pytree(grads)
        grads_flat += curr_flat_grads
        loss += curr_loss

    loss = loss / seq_len
    return loss, unravel_fn(grads_flat)


if __name__ == "__main__":
    # グローバルなシードを設定
    SEED = 0
    np.random.seed(SEED)
    rng_key = jax.random.PRNGKey(SEED)

    seq_len = 30
    batch_size = 1
    hidden_size = 20
    num_steps = 100

    # ダミーなランダムデータとして入力データとターゲットデータを作成
    data_x = np.random.randn(seq_len, batch_size, hidden_size).astype(np.float32)
    data_y = np.random.randn(seq_len, batch_size, hidden_size).astype(np.float32)

    # JAXのPRNGキーを分割して使用
    rng_key, init_key = jax.random.split(rng_key)

    # モデルの初期化
    model_cell_bptt = nn.SimpleCell(features=hidden_size)
    model_cell_rtrl = RtrlCell(hidden_size=hidden_size)

    params_bptt = model_cell_bptt.init(
        init_key,
        jnp.ones((batch_size, hidden_size)),
        jnp.ones((batch_size, hidden_size)),
    )
    params_rtrl = model_cell_rtrl.init(
        init_key,
        model_cell_rtrl.initialize_state(batch_size, hidden_size, hidden_size),
        jnp.ones((batch_size, hidden_size)),
    )

    # 訓練状態の作成
    state_bptt = train_state.TrainState.create(
        apply_fn=model_cell_bptt.apply, params=params_bptt, tx=optax.adam(1e-3)
    )
    state_rtrl = train_state.TrainState.create(
        apply_fn=model_cell_rtrl.apply, params=params_rtrl, tx=optax.adam(1e-3)
    )

    print("params_rtrl:")
    print_dict_tree(params_rtrl)
    print("params_bptt:")
    print_dict_tree(params_bptt)

    # copy params from bptt to rtrl
    params_rtrl["params"]["RtrlRNNCellFwd_0"] = params_bptt["params"]

    loss_bptt, grads_bptt = bptt_grads(state_bptt, data_x, data_y)
    loss_rtrl, grads_rtrl = rtrl_grads(state_rtrl, data_x, data_y)

    print("BPTT Loss:", loss_bptt)
    print("RTRL Loss:", loss_rtrl)

    loss_diff = jnp.abs(loss_bptt - loss_rtrl)
    print(f"{loss_diff=}")

    grads_bptt_W = grads_bptt["params"]["i"]["kernel"]
    grads_rtrl_W = grads_rtrl["params"]["RtrlRNNCellFwd_0"]["i"]["kernel"]
    grads_diff_W = jnp.abs(grads_bptt_W - grads_rtrl_W)
    grads_diff_W = jnp.mean(grads_diff_W)
    print(f"{grads_diff_W=}")

    grads_bptt_B = grads_bptt["params"]["i"]["bias"]
    grads_rtrl_B = grads_rtrl["params"]["RtrlRNNCellFwd_0"]["i"]["bias"]
    grads_diff_B = jnp.abs(grads_bptt_B - grads_rtrl_B)
    grads_diff_B = jnp.mean(grads_diff_B)
    print(f"{grads_diff_B=}")

    grads_bptt_R = grads_bptt["params"]["h"]["kernel"]
    grads_rtrl_R = grads_rtrl["params"]["RtrlRNNCellFwd_0"]["h"]["kernel"]
    grads_diff_R = jnp.abs(grads_bptt_R - grads_rtrl_R)
    grads_diff_R = jnp.mean(grads_diff_R)
    print(f"{grads_diff_R=}")
