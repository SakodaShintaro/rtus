import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
import numpy as np


# RNN Model Definition
class RNNModel(nn.Module):
    hidden_size: int
    output_dim: int

    @nn.compact
    def __call__(self, x, carry):
        rnn = nn.SimpleCell(features=self.hidden_size)
        carry, hidden = rnn(carry, x)
        y = nn.Dense(features=self.output_dim)(hidden)
        return y, carry


def loss_fn(params, model, x, y):
    # x shape: [seq_len, batch_size, input_dim]
    batch_size = x.shape[1]

    # 最初のキャリー状態を初期化
    # SimpleCell用の初期状態を作成（0で初期化されたベクトル）
    # モデルのパラメータから隠れ層のサイズを取得
    hidden_size = params["params"]["SimpleCell_0"]["h"]["kernel"].shape[0]
    initial_carry = jnp.zeros((batch_size, hidden_size))

    def step_fn(carry, x_t):
        # x_t shape: [batch_size, input_dim]
        y_pred, new_carry = model(params, x_t, carry)
        return new_carry, y_pred

    # scanでシーケンス全体に対して処理を適用
    _, y_pred = jax.lax.scan(step_fn, initial_carry, x)
    # y_pred shape: [seq_len, batch_size, hidden_size]

    # MSE損失を計算
    loss = jnp.mean((y_pred - y) ** 2)
    return loss


# Training Setup
@jax.jit
def train_step(state, batch_x, batch_y):
    loss, grads = jax.value_and_grad(loss_fn)(
        state.params, state.apply_fn, batch_x, batch_y
    )
    state = state.apply_gradients(grads=grads)
    return state, loss


if __name__ == "__main__":
    seq_len = 10
    batch_size = 64
    input_dim = 5
    hidden_size = 20
    num_steps = 100

    # 入力データとターゲットデータの作成
    data_x = np.random.randn(seq_len, batch_size, input_dim).astype(np.float32)
    data_y = np.random.randn(seq_len, batch_size, input_dim).astype(np.float32)

    # モデルの初期化
    model = RNNModel(hidden_size=hidden_size, output_dim=input_dim)
    # 単一時点のバッチデータでモデルを初期化
    params = model.init(
        jax.random.PRNGKey(0),
        jnp.ones((batch_size, input_dim)),
        jnp.ones((batch_size, hidden_size)),
    )

    # 訓練状態の作成
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(1e-3)
    )

    # トレーニングループ
    for step in range(num_steps):
        state, loss = train_step(state, data_x, data_y)
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1:03d}/{num_steps}, Loss: {loss:.6f}")
