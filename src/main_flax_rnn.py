import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
import numpy as np


# RNN Model Definition
class RNNModel(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x, carry=None):
        rnn = nn.SimpleCell(features=self.hidden_size)
        if carry is None:
            carry = rnn.initialize_carry(jax.random.PRNGKey(0), (x.shape[0],))
        carry, y = rnn(carry, x)
        return y, carry


def loss_fn(params, model, x, y):
    def step_fn(x_t, carry):
        y_pred, carry = model.apply({"params": params}, x_t, carry)
        return carry, y_pred

    carry = None
    _, y_pred = jax.lax.scan(step_fn, carry, x)
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
    # Data Generation
    seq_length = 10
    num_samples = 1000
    input_dim = 5
    hidden_size = 20

    data_x = np.random.randn(num_samples, seq_length, input_dim).astype(np.float32)
    data_y = np.sum(data_x, axis=2)  # Simple sum as target

    model = RNNModel(hidden_size=hidden_size)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((seq_length, input_dim)))
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(1e-3)
    )

    # Training Loop
    num_epochs = 10
    batch_size = 32
    for epoch in range(num_epochs):
        for i in range(0, num_samples, batch_size):
            batch_x = data_x[i : i + batch_size]
            batch_y = data_y[i : i + batch_size]
            state, loss = train_step(state, batch_x, batch_y)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
