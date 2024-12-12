import jax.numpy as jnp
from flax import linen as nn


class BiasForce(nn.Module):
    def setup(self):
        self.output_dim = 2
        self.input_dim = 2 + 1

        self.dense1 = nn.Dense(32)
        self.dense2 = nn.Dense(16)
        self.dense3 = nn.Dense(self.output_dim)

    def __call__(self, pos, target):
        dist = jnp.linalg.norm(pos - target, axis=-1, keepdims=True)
        pos_ = jnp.concatenate([pos, dist], axis=-1)

        x = self.dense1(pos_.reshape(-1, self.input_dim))
        x = nn.relu(x)
        x = self.dense2(x)
        x = nn.relu(x)
        out = self.dense3(x)

        force = out.reshape(*pos.shape)
        return force
