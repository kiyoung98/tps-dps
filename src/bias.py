import jax
import jax.numpy as jnp
from typing import Tuple
from jax.nn import softplus
from flax import linen as nn
from utils.utils import kabsch_align


class BiasForce(nn.Module):
    hidden_dims: Tuple[int, ...]
    bias: str = "pot"

    @nn.compact
    def __call__(self, pos, target):
        pos_ = pos.reshape(-1, pos.shape[-1] // 3, 3)

        target_ = target.reshape(-1, target.shape[-1] // 3, 3)
        target_ = jnp.tile(target_, (pos_.shape[0], 1, 1))

        R, t = kabsch_align(pos_, target_)

        if self.bias == "force":
            force = self.forward(pos_, target_, R, t)
            force = jnp.matmul(force, R)
            force = force.reshape(*pos.shape)
        elif self.bias == "pot":
            force = -jax.grad(lambda x: self.forward(x, target_, R, t).sum())(
                pos_,
            )
            force = force.reshape(*pos.shape)
        elif self.bias == "scale":
            out = self.forward(pos_, target_, R, t)
            target_align = jnp.matmul(target_ - t, R)
            force = softplus(out) * (target_align - pos_)
            force = force.reshape(*pos.shape)
        return force

    def forward(self, pos, target, R, t):
        pos_align = jnp.matmul(pos, R.transpose(0, 2, 1)) + t
        dist = jnp.linalg.norm(pos_align - target, axis=-1, keepdims=True)
        x = jnp.concatenate([pos_align, dist], axis=-1)
        x = x.reshape(x.shape[0], -1)
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        if self.bias == "force":
            x = nn.Dense(pos.shape[-1] * pos.shape[-2])(x)
            x = x.reshape(x.shape[0], pos.shape[-2], pos.shape[-1])
        elif self.bias == "pot":
            x = nn.Dense(1)(x)
        elif self.bias == "scale":
            x = nn.Dense(pos.shape[-2])(x)
            x = x.reshape(x.shape[0], pos.shape[-2], 1)
        return x
