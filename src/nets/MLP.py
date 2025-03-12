from typing import Callable, Iterable
import flax.linen as nn


class MLP(nn.Module):
    hiddens: Iterable[int]
    act: Callable = nn.relu
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        for size in self.hiddens[:-1]:
            x = nn.Dense(size)(x)
            x = self.act(x)
        return nn.Dense(self.hiddens[-1])(x)
