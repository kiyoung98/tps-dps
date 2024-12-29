import jax
import jax.numpy as jnp
from jax.nn import softplus
from flax import linen as nn


class BiasForce(nn.Module):
    bias: str = "pot"
    hidden_dim: int = 32

    @nn.compact
    def __call__(self, pos, target):

        if self.bias == "force":
            out = self.forward(pos, target)
            force = out.reshape(*pos.shape)
        elif self.bias == "pot":
            force = -jax.grad(lambda x: self.forward(x, target).sum())(
                pos,
            ).reshape(*pos.shape)
        elif self.bias == "scale":
            out = self.forward(pos, target)
            force = softplus(out) * (target - pos)
        return force

    def forward(self, pos, target):
        dist = jnp.linalg.norm(pos - target, axis=-1, keepdims=True)
        x = jnp.concatenate([pos, dist], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        if self.bias == "force":
            x = nn.Dense(2)(x)
        elif self.bias == "pot":
            x = nn.Dense(1)(x)
        elif self.bias == "scale":
            x = nn.Dense(2)(x)
        return x
