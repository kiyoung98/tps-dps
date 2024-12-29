import jax
import optax
import jax.numpy as jnp
from bias import BiasForce


class DiffusionPathSampler:
    def __init__(self, args, mds, key):
        self.policy = BiasForce(args.bias, args.hidden_dim)
        self.replay = ReplayBuffer(args)
        self.target_measure = TargetMeasure(args, mds)

        self.timestep = args.timestep

        self.params = self.policy.init(
            key,
            jnp.zeros(
                (args.batch_size, args.num_steps, 2),
            ),
            mds.target_position,
        )

        self.optimizer = optax.adam(args.policy_lr)
        self.opt_state = self.optimizer.init(self.params)

        @jax.jit
        def loss_fn(params, positions, forces, log_tm):
            biases = self.policy.apply(params, positions, mds.target_position)
            means = positions + (forces + biases) * self.timestep
            log_bpm = mds.log_prob(positions[:, 1:] - means[:, :-1]).mean((1, 2))

            log_z = (log_tm - log_bpm).mean()
            loss = ((log_z + log_bpm - log_tm) ** 2).mean()
            return loss

        self.grad_fn = jax.value_and_grad(loss_fn)

    def sample(self, args, mds, std, key):
        def step(carry, key):
            position, force = carry
            bias = self.policy.apply(self.params, position, mds.target_position)
            position = (
                position
                + (force + bias) * args.timestep
                + std * jax.random.normal(key, position.shape)
            )
            force = -mds.dUdx(position)
            return (position, force), (position, force, mds.U(position))

        position = jnp.tile(mds.start_position, (args.num_samples, 1))
        force = -mds.dUdx(position)
        keys = jax.random.split(key, args.num_steps)
        positions, forces, potentials = jax.lax.scan(step, (position, force), keys)[1]
        positions = jnp.swapaxes(positions, 0, 1)
        forces = jnp.swapaxes(forces, 0, 1)
        potentials = jnp.swapaxes(potentials, 0, 1)
        log_tm = self.target_measure(positions, forces)
        self.replay.add((positions, forces, log_tm))

        return positions, potentials

    def train(self, args, key):
        def train_step(carry, key):
            params, opt_state = carry
            positions, forces, log_tm = self.replay.sample(key)
            loss, grads = self.grad_fn(params, positions, forces, log_tm)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        keys = jax.random.split(key, args.trains_per_rollout)
        (self.params, self.opt_state), losses = jax.lax.scan(
            train_step,
            (self.params, self.opt_state),
            keys,
            length=args.trains_per_rollout,
        )
        loss = losses.mean()
        return loss


class ReplayBuffer:
    def __init__(self, args):
        self.positions = jnp.zeros(
            (args.buffer_size, args.num_steps, 2),
        )
        self.forces = jnp.zeros(
            (args.buffer_size, args.num_steps, 2),
        )
        self.log_reward = jnp.zeros(args.buffer_size)

        self.idx = 0
        self.batch_size = args.batch_size
        self.num_samples = args.num_samples
        self.buffer_size = args.buffer_size

    def add(self, data):
        indices = jnp.arange(self.idx, self.idx + self.num_samples) % self.buffer_size
        self.idx += self.num_samples

        self.positions = self.positions.at[indices].set(data[0])
        self.forces = self.forces.at[indices].set(data[1])
        self.log_reward = self.log_reward.at[indices].set(data[2])

    def sample(self, key):
        indices = jax.random.randint(
            key, (self.batch_size,), 0, min(self.idx, self.buffer_size)
        )
        return (
            self.positions[indices],
            self.forces[indices],
            self.log_reward[indices],
        )


class TargetMeasure:
    def __init__(self, args, mds):
        self.sigma = args.sigma
        self.log_prob = mds.log_prob
        self.timestep = args.timestep
        self.target_position = mds.target_position

    def __call__(self, positions, forces):
        log_upm = self.unbiased_path_measure(positions, forces)
        log_ri = self.log_relaxed_indicator(positions[:, -1])

        log_reward = log_upm + log_ri
        return log_reward

    def unbiased_path_measure(self, positions, forces):
        means = positions + forces * self.timestep
        log_upm = self.log_prob(positions[:, 1:] - means[:, :-1]).mean((1, 2))
        return log_upm

    def log_relaxed_indicator(self, final_position):
        log_rbf = (
            -0.5
            / self.sigma**2
            * ((final_position - self.target_position) ** 2).mean(-1)
        )
        return log_rbf
