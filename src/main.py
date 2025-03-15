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
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
    n_params = flat_params.shape[0]
    batch_size = batch_x.shape[1]
    # SimpleCellの重みサイズから隠れ状態の次元を取得
    hidden_size = params["params"]["SimpleCell_0"]["h"]["kernel"].shape[0]

    Wh = params["params"]["SimpleCell_0"]["h"]["kernel"]
    Wi = params["params"]["SimpleCell_0"]["i"]["kernel"]
    Wi_b = params["params"]["SimpleCell_0"]["i"]["bias"]

    Wy = params["params"]["Dense_0"]["kernel"]
    Wy_b = params["params"]["Dense_0"]["bias"]

    # 最初のキャリー状態を初期化
    h_t = jnp.zeros((batch_size, hidden_size))

    # RNNの更新式は
    # h(t+1) = \tanh(Wi x + bi + Wh h(t))

    # 感度行列 : 隠れ状態に対するパラメータの偏微分を保持 (batch, hidden_size, n_params)
    S = jnp.zeros((batch_size, hidden_size, n_params))

    grad_flat = jnp.zeros(n_params)
    seq_len = batch_x.shape[0]

    loss = 0.0

    # 各ステップの勾配和
    # dL_{total}(1,T)/dW = \sum _ {t=1} ^ {T} dL_{t}/dW
    # dL_{t}/dW = \sum _ {k=1} ^{N} (dL_{t}/dh_{t} * dh_{t}/dW)
    # つまり以下の2つで勾配が計算できる
    # (a) dL_{t}/dh_{t} : このステップでの損失に対する隠れ状態の勾配
    #     これは普通に計算できる
    # (b) dh_{t}/dW : 感度行列(S(t))
    #     これは各ステップで再帰的に計算できる
    #     要するにh(t-1)を定数だと思って、かつS(t)をかけたものの勾配を求める

    for t in range(seq_len):
        x_t = batch_x[t]
        y_t_ref = batch_y[t]

        # print(f"{Wi.shape=}")  # (5, 20)
        # print(f"{x_t.shape=}")  # (1, 5)
        # print(f"{Wi_b.shape=}")  # (20,)
        # print(f"{Wh.shape=}")  # (20, 20)
        # print(f"{h.shape=}")  # (1, 20)
        # print(f"{Wy.shape=}")  # (20, 5)
        # print(f"{Wy_b.shape=}")  # (5,)
        # print(f"{y_t_ref.shape=}")  # (1, 5)

        s_t = jnp.dot(x_t, Wi) + Wi_b + jnp.dot(h_t, Wh)
        h_t = jnp.tanh(s_t)
        y_t_sub = jnp.dot(h_t, Wy) + Wy_b
        loss_t = jnp.mean((y_t_sub - y_t_ref) ** 2)
        loss += loss_t

    # 平均損失を返す
    loss = loss / seq_len

    return loss, unravel_fn(grad_flat)


if __name__ == "__main__":
    # グローバルなシードを設定
    SEED = 0
    np.random.seed(SEED)
    rng_key = jax.random.PRNGKey(SEED)

    seq_len = 10
    batch_size = 1
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

    loss_diff = jnp.abs(loss_bptt - loss_rtrl)
    print(f"{loss_diff=}")

    grads_bptt, _ = jax.flatten_util.ravel_pytree(grads_bptt)
    grads_rtrl, _ = jax.flatten_util.ravel_pytree(grads_rtrl)
    grads_diff = jnp.abs(grads_bptt - grads_rtrl)
    grads_diff_mean = jnp.mean(grads_diff)
    print(f"{grads_diff_mean=}")
