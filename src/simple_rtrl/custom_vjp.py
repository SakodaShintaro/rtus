import jax
import jax.numpy as jnp
from flax.training import train_state
import flax.linen as nn
import flax
import optax
import numpy as np
from functools import partial

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


class RtrlRNNCellFwd(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, carry, x):
        prev_h, prev_sensitivity_matrix = carry
        rnn = nn.SimpleCell(features=self.hidden_size)

        # update hidden state
        curr_h, curr_out = rnn(prev_h, x)

        # update sensitivity matrix (TODO)
        curr_sensitivity_matrix = prev_sensitivity_matrix

        return (curr_h, curr_sensitivity_matrix), curr_out


class RtrlCell(nn.Module):
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
        return carry, hidden

    @staticmethod
    def initialize_state(batch_size, d_rec, d_input):
        hidden_init = jnp.zeros((batch_size, d_rec))
        memory_grad_init = ()
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
    hidden_size = params["params"]["RtrlRNNCellFwd_0"]["SimpleCell_0"]["h"][
        "kernel"
    ].shape[0]
    carry = (
        jnp.zeros((batch_size, hidden_size)),
        jnp.zeros((batch_size, hidden_size)),
    )

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
    params_rtrl["params"]["RtrlRNNCellFwd_0"]["SimpleCell_0"] = params_bptt["params"]

    loss_bptt, grads_bptt = bptt_grads(state_bptt, data_x, data_y)
    loss_rtrl, grads_rtrl = rtrl_grads(state_rtrl, data_x, data_y)

    print("BPTT Loss:", loss_bptt)
    print("RTRL Loss:", loss_rtrl)

    loss_diff = jnp.abs(loss_bptt - loss_rtrl)
    print(f"{loss_diff=}")

    print("grads_rtrl:")
    print_dict_tree(grads_rtrl)
    print("grads_bptt:")
    print_dict_tree(grads_bptt)

    grads_bptt_W = grads_bptt["params"]["i"]["kernel"]
    grads_rtrl_W = grads_rtrl["params"]["RtrlRNNCellFwd_0"]["SimpleCell_0"]["i"][
        "kernel"
    ]
    grads_diff_W = jnp.abs(grads_bptt_W - grads_rtrl_W)
    grads_diff_W = jnp.mean(grads_diff_W)
    print(f"{grads_diff_W=}")

    grads_bptt_B = grads_bptt["params"]["i"]["bias"]
    grads_rtrl_B = grads_rtrl["params"]["RtrlRNNCellFwd_0"]["SimpleCell_0"]["i"]["bias"]
    grads_diff_B = jnp.abs(grads_bptt_B - grads_rtrl_B)
    grads_diff_B = jnp.mean(grads_diff_B)
    print(f"{grads_diff_B=}")

    grads_bptt_R = grads_bptt["params"]["h"]["kernel"]
    grads_rtrl_R = grads_rtrl["params"]["RtrlRNNCellFwd_0"]["SimpleCell_0"]["h"][
        "kernel"
    ]
    grads_diff_R = jnp.abs(grads_bptt_R - grads_rtrl_R)
    grads_diff_R = jnp.mean(grads_diff_R)
    print(f"{grads_diff_R=}")
