import jax
import jax.numpy as jnp
from jax import jit, random
import optax
from typing import NamedTuple, Tuple, Optional


class RNNParams(NamedTuple):
    W_h: jnp.ndarray  # 隠れ状態の重み
    W_x: jnp.ndarray  # 入力の重み
    W_y: jnp.ndarray  # 出力の重み
    b_h: jnp.ndarray  # 隠れ状態のバイアス
    b_y: jnp.ndarray  # 出力のバイアス


def init_params(
    key: jnp.ndarray, hidden_size: int, input_size: int, output_size: int
) -> RNNParams:
    k1, k2, k3 = random.split(key, 3)

    # Xavier初期化を使用してパラメータを初期化
    W_h = random.normal(k1, (hidden_size, hidden_size)) * jnp.sqrt(
        2.0 / (hidden_size + hidden_size)
    )
    W_x = random.normal(k2, (input_size, hidden_size)) * jnp.sqrt(
        2.0 / (input_size + hidden_size)
    )
    W_y = random.normal(k3, (hidden_size, output_size)) * jnp.sqrt(
        2.0 / (hidden_size + output_size)
    )
    b_h = jnp.zeros((hidden_size,))
    b_y = jnp.zeros((output_size,))

    return RNNParams(W_h=W_h, W_x=W_x, W_y=W_y, b_h=b_h, b_y=b_y)


def rnn_step(params: RNNParams, h_prev: jnp.ndarray, x_t: jnp.ndarray) -> jnp.ndarray:
    h_t = jnp.tanh(jnp.dot(x_t, params.W_x) + jnp.dot(h_prev, params.W_h) + params.b_h)
    return h_t


def rnn_forward(
    params: RNNParams, inputs: jnp.ndarray, h_0: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    seq_len, batch_size, input_size = inputs.shape
    _, hidden_size = params.W_x.shape

    # 初期隠れ状態が与えられていない場合は0で初期化
    if h_0 is None:
        h_0 = jnp.zeros((batch_size, hidden_size))

    # 各時間ステップでRNNを実行
    def scan_fn(
        h_prev: jnp.ndarray, x_t: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        h_t = rnn_step(params, h_prev, x_t)
        y_t = jnp.dot(h_t, params.W_y) + params.b_y
        return h_t, y_t

    # JAXのscanを使って系列を処理
    final_h, outputs = jax.lax.scan(scan_fn, h_0, inputs)

    return outputs, final_h


def mse_loss(
    params: RNNParams,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    h_0: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    predictions, _ = rnn_forward(params, inputs, h_0)
    return jnp.mean((predictions - targets) ** 2)


@jit
def train_step(
    params: RNNParams,
    opt_state: optax.OptState,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
) -> Tuple[RNNParams, optax.OptState, jnp.ndarray]:
    def loss_fn(p: RNNParams) -> jnp.ndarray:
        return mse_loss(p, inputs, targets)

    loss, grads = jax.value_and_grad(loss_fn)(params)

    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss


if __name__ == "__main__":
    # パラメータ設定
    hidden_size: int = 32
    input_size: int = 10
    output_size: int = 5
    batch_size: int = 16
    seq_len: int = 20
    learning_rate: float = 0.01
    num_steps: int = 100

    # パラメータ初期化
    key: jnp.ndarray = random.PRNGKey(0)
    key, data_key = random.split(key)
    params: RNNParams = init_params(key, hidden_size, input_size, output_size)

    # オプティマイザーの設定
    optimizer: optax.GradientTransformation = optax.adam(learning_rate)
    opt_state: optax.OptState = optimizer.init(params)

    # ランダムな入力データと目標データを生成
    data_key, target_key = random.split(key)
    inputs = random.normal(data_key, (seq_len, batch_size, input_size))
    targets = random.normal(target_key, (seq_len, batch_size, output_size))

    # 訓練ループ
    for step in range(num_steps):
        params, opt_state, loss = train_step(params, opt_state, inputs, targets)
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1:03d}/{num_steps}, Loss: {loss:.6f}")
