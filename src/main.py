import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
import numpy as np
from functools import partial


class RNNModel(nn.Module):
    hidden_size: int
    output_dim: int

    @nn.compact
    def __call__(self, carry, x):
        rnn = nn.SimpleCell(features=self.hidden_size)
        carry, hidden = rnn(carry, x)
        y = nn.Dense(features=self.output_dim)(hidden)
        return carry, y


def bptt_loss_fn(params, model, x, y):
    # x shape: [seq_len, batch_size, input_dim]
    batch_size = x.shape[1]

    # 最初のキャリー状態を初期化
    hidden_size = params["params"]["SimpleCell_0"]["h"]["kernel"].shape[0]
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
    model = state.apply_fn
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
    n_params = flat_params.shape[0]
    batch_size = batch_x.shape[1]
    # SimpleCellの重みサイズから隠れ状態の次元を取得
    hidden_size = params["params"]["SimpleCell_0"]["h"]["kernel"].shape[0]
    c = jnp.zeros((batch_size, hidden_size))
    # 各サンプルについて隠れ状態のパラメータに対する微分を保持 (batch, hidden_size, n_params)
    S = jnp.zeros((batch_size, hidden_size, n_params))
    grad_flat = jnp.zeros(n_params)
    seq_len = batch_x.shape[0]

    loss = 0.0

    for t in range(seq_len):
        xt = batch_x[t]

        # 現在の時刻の更新関数
        def step_fn(theta, c):
            p = unravel_fn(theta)
            new_c, y_out = model(p, c, xt)
            return new_c, y_out

        new_c, y_out = step_fn(flat_params, c)

        # cからnew_cへの微分 (A) を各サンプルごとに求める
        def f_c(c_sample):
            new_c, _ = step_fn(flat_params, c_sample)
            return new_c

        A = jax.vmap(jax.jacfwd(f_c))(c)  # shape: (batch, hidden_size, hidden_size)

        # パラメータからnew_cへの微分 (B) を各サンプルごとに求める
        def f_theta(c_sample):
            def inner(theta):
                new_c, _ = step_fn(theta, c_sample)
                return new_c

            return jax.jacfwd(inner)(flat_params)

        B = jax.vmap(f_theta)(c)  # shape: (batch, hidden_size, n_params)
        # 感度行列更新: S = A @ S + B
        print(type(A), type(S), type(B))
        print(f"{A.shape=}, {S.shape=}, {B.shape=}")
        S = jnp.einsum("bij,bjk->bik", A, S) + B

        # y出力について，cとパラメータの直接微分を求める
        def f_y(c_sample):
            _, y_out = step_fn(flat_params, c_sample)
            return y_out

        dYdC = jax.vmap(jax.jacfwd(f_y))(c)  # (batch, output_dim, hidden_size)

        def f_y_theta(c_sample):
            def inner(theta):
                _, y_out = step_fn(theta, c_sample)
                return y_out

            return jax.jacfwd(inner)(flat_params)

        dYdTheta = jax.vmap(f_y_theta)(c)  # (batch, output_dim, n_params)
        # 全微分: dY/dθ = (dY/dc)·S + dY/dθ(直接)
        dYdTheta_total = jnp.einsum("bij,bjk->bik", dYdC, S) + dYdTheta

        # MSE損失の微分 (全要素平均なのでスケールも合わせる)
        output_dim = y_out.shape[1]
        loss += jnp.mean((y_out - batch_y[t]) ** 2)
        dloss_dy = 2 * (y_out - batch_y[t]) / (seq_len * batch_size * output_dim)
        # 各サンプル・出力軸について鎖率則を適用し，勾配寄与を合成
        grad_t = jnp.einsum("bi,bik->k", dloss_dy, dYdTheta_total)
        grad_flat += grad_t
        c = new_c
    return loss, grad_flat

if __name__ == "__main__":
    # グローバルなシードを設定
    SEED = 0
    np.random.seed(SEED)
    rng_key = jax.random.PRNGKey(SEED)

    seq_len = 10
    batch_size = 64
    input_dim = 5
    hidden_size = 20
    num_steps = 100

    # ダミーなランダムデータとして入力データとターゲットデータを作成
    data_x = np.random.randn(seq_len, batch_size, input_dim).astype(np.float32)
    data_y = np.random.randn(seq_len, batch_size, input_dim).astype(np.float32)

    # JAXのPRNGキーを分割して使用
    rng_key, init_key = jax.random.split(rng_key)

    # モデルの初期化
    model = RNNModel(hidden_size=hidden_size, output_dim=input_dim)
    params = model.init(
        init_key,
        jnp.ones((batch_size, hidden_size)),
        jnp.ones((batch_size, input_dim)),
    )

    # 訓練状態の作成
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(1e-3)
    )

    loss_bptt, grads_bptt = bptt_grads(state, data_x, data_y)
    loss_rtrl, grads_rtrl = rtrl_grads(state, data_x, data_y)

    print("BPTT Loss:", loss_bptt)
    print("RTRL Loss:", loss_rtrl)
