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


@jax.jit
def dtanh(x):
    return 1 - jnp.tanh(x) ** 2


def rtrl_grads(state, batch_x, batch_y):
    params = state.params
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
    n_params = flat_params.shape[0]
    batch_size = batch_x.shape[1]
    hidden_size = params["params"]["SimpleCell_0"]["h"]["kernel"].shape[0]

    R = params["params"]["SimpleCell_0"]["h"]["kernel"]
    W = params["params"]["SimpleCell_0"]["i"]["kernel"]
    B = params["params"]["SimpleCell_0"]["i"]["bias"]

    Wy = params["params"]["Dense_0"]["kernel"]
    Wy_b = params["params"]["Dense_0"]["bias"]

    # 最初のキャリー状態を初期化
    h_t = jnp.zeros((batch_size, hidden_size))

    # RNNの更新式は
    #   h(t+1) = \tanh(W x + B + R h(t))

    # 感度行列 : 隠れ状態に対するパラメータの偏微分を保持 (batch, hidden_size, n_params)
    S_W = jnp.zeros((batch_size, hidden_size, *W.shape))
    S_R = jnp.zeros((batch_size, hidden_size, *R.shape))

    grad_structured = unravel_fn(jnp.zeros(n_params))
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

    for t in range(seq_len):
        print(f"{t=}")
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

        s_t = jnp.dot(x_t, W) + B + jnp.dot(h_t, R)
        h_t = jnp.tanh(s_t)
        y_t_sub = jnp.dot(h_t, Wy) + Wy_b
        loss_t = jnp.mean((y_t_sub - y_t_ref) ** 2)
        loss += loss_t

        dl_dyt = 2 * (y_t_sub - y_t_ref)
        dl_dh = jnp.dot(dl_dyt, Wy.T)
        dl_ds = dl_dh * (1 - h_t**2)
        print(f"{dl_ds.shape=}")

        # 感度行列の更新
        S_W = np.array(S_W)
        S_R = np.array(S_R)

        print("S_W")
        new_S_W = np.zeros_like(S_W)
        for k in range(hidden_size):
            for i in range(S_W.shape[-2]):
                for j in range(S_W.shape[-1]):
                    new_S_W[:, k, i, j] = x_t[:, i] * (k == j)
                    new_S_W[:, k, i, j] += np.einsum(
                        "n,bn,bn->b", R[k], dtanh(s_t), S_W[:, :, i, j]
                    )
                    # for n in range(hidden_size):
                    #     new_S_W[:, k, i, j] += (
                    #         R[k, n] * dtanh(s_t[:, n]) * S_W[:, n, i, j]
                    #     )
        print("S_R")
        new_S_R = np.zeros_like(S_R)
        for k in range(hidden_size):
            for i in range(S_R.shape[-2]):
                for j in range(S_R.shape[-1]):
                    new_S_R[:, k, i, j] = np.sum(h_t, axis=1) * (i == k)
                    new_S_R[:, k, i, j] += np.einsum(
                        "n,bn,bn->b", R[k], dtanh(s_t), S_R[:, :, i, j]
                    )

                    # for n in range(hidden_size):
                    #     new_S_R[:, k, i, j] += (
                    #         (h_t[:, n] * (i == k and j == n))
                    #         + (R[k, n] * dtanh(s_t[:, n]) * S_R[:, n, i, j])
                    #     )

        # S_Wの更新：new_S_W[b,k,i,j] = x_t[b,i]*δ_{k,j} + Σ_n R[k,n]*dtanh(s_t[b,n])*S_W[b,n,i,j]
        # new_S_W = jnp.einsum("bi,kj->bkij", x_t, jnp.eye(hidden_size)) + jnp.einsum(
        #     "kn,bn,bnij->bkij", R, dtanh(s_t), S_W
        # )

        # S_Rの更新：new_S_R[b,k,i,j] = δ_{k,i}*tanh(s_t[b,j]) + Σ_n R[k,n]*dtanh(s_t[b,n])*S_R[b,n,i,j]
        # new_S_R = jnp.eye(hidden_size)[None, :, :, None] * jnp.tanh(s_t)[
        #     :, None, None, :
        # ] + jnp.einsum("kn,bn,nbij->bkij", R, dtanh(s_t), S_R)

        S_W = jnp.array(new_S_W)
        S_R = jnp.array(new_S_R)

        print(f"{S_W.shape=}")
        print(f"{S_R.shape=}")

        curr_grad_W = jnp.einsum("bh,bhij->bij", dl_ds, S_W)
        curr_grad_R = jnp.einsum("bh,bhij->bij", dl_ds, S_R)

        grad_structured["params"]["SimpleCell_0"]["i"]["kernel"] += curr_grad_W
        grad_structured["params"]["SimpleCell_0"]["h"]["kernel"] += curr_grad_R

    # 平均損失を返す
    loss = loss / seq_len

    return loss, grad_structured


if __name__ == "__main__":
    # グローバルなシードを設定
    SEED = 0
    np.random.seed(SEED)
    rng_key = jax.random.PRNGKey(SEED)

    seq_len = 2
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

    grads_bptt_W = grads_bptt["params"]["SimpleCell_0"]["i"]["kernel"]
    grads_rtrl_W = grads_rtrl["params"]["SimpleCell_0"]["i"]["kernel"]
    grads_diff_W = jnp.abs(grads_bptt_W - grads_rtrl_W)
    grads_diff_W = jnp.mean(grads_diff_W)
    print(f"{grads_diff_W=}")

    grads_bptt_R = grads_bptt["params"]["SimpleCell_0"]["h"]["kernel"]
    grads_rtrl_R = grads_rtrl["params"]["SimpleCell_0"]["h"]["kernel"]
    grads_diff_R = jnp.abs(grads_bptt_R - grads_rtrl_R)
    grads_diff_R = jnp.mean(grads_diff_R)
    print(f"{grads_diff_R=}")
