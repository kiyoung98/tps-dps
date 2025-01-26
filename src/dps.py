import jax
import optax
import jax.numpy as jnp
from bias import BiasForce
from utils.utils import kabsch_align


class DiffusionPathSampler:
    def __init__(self, args, sys, key):
        self.replay = ReplayBuffer(args, sys)
        self.target_measure = TargetMeasure(args, sys)
        self.policy = BiasForce(args.hidden_dims, args.bias)

        policy_params = self.policy.init(
            key,
            jnp.zeros(
                (args.batch_size, args.num_steps, sys.dim),
            ),
            sys.B,
        )
        log_z = jnp.array(0.0)

        self.optimizer = optax.chain(
            optax.multi_transform(
                {
                    "policy": optax.adam(args.policy_lr),
                    "log_z": optax.adam(args.log_z_lr),
                },
                param_labels={"policy": "policy", "log_z": "log_z"},
            ),
            optax.clip_by_global_norm(args.max_grad_norm),
        )

        self.params = {"policy": policy_params, "log_z": log_z}
        self.opt_state = self.optimizer.init(self.params)

        @jax.jit
        def loss_fn(params, positions, forces, log_tm):
            biases = self.policy.apply(params["policy"], positions, sys.B)
            velocities = (positions[:, 1:] - positions[:, :-1]) / args.timestep

            means = (1 - args.gamma * args.timestep) * velocities + (
                forces[:, :-1] + biases[:, :-1]
            ) * args.timestep / sys.mass
            log_bpm = sys.log_prob(velocities[:, 1:] - means[:, :-1]).mean((1, 2))
            loss = jnp.mean((params["log_z"] + log_bpm - log_tm) ** 2)
            return loss

        self.grad_fn = jax.value_and_grad(loss_fn)

    def sample(self, args, sys, std, key):
        def step(carry, key):
            position, velocity, force = carry
            bias = self.policy.apply(self.params["policy"], position, sys.B)
            velocity = (
                (1 - args.gamma * args.timestep) * velocity
                + (force + bias) * args.timestep / sys.mass
                + std * jax.random.normal(key, velocity.shape)
            )
            position = position + velocity * args.timestep
            return (position, velocity, -sys.dUdx(position)), (
                position,
                -sys.dUdx(position),
                sys.U(position),
            )

        keys = jax.random.split(key, args.num_steps)

        position = jnp.tile(sys.A, (args.num_samples, 1))
        carry = (position, jnp.zeros_like(position), -sys.dUdx(position))

        positions, forces, potentials = jax.lax.scan(step, carry, keys)[1]

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
            updates, opt_state = self.optimizer.update(
                grads, opt_state, param_labels=params
            )
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        sample_keys = jax.random.split(key, args.trains_per_rollout)
        (self.params, self.opt_state), losses = jax.lax.scan(
            train_step,
            (self.params, self.opt_state),
            sample_keys,
            length=args.trains_per_rollout,
        )
        loss = losses.mean()

        loss = jnp.mean(jnp.array(losses))
        return loss


class ReplayBuffer:
    def __init__(self, args, sys):
        self.positions = jnp.zeros(
            (args.buffer_size, args.num_steps, sys.dim),
        )
        self.forces = jnp.zeros(
            (args.buffer_size, args.num_steps, sys.dim),
        )
        self.log_reward = jnp.zeros(args.buffer_size)

        self.idx = 0
        self.batch_size = args.batch_size
        self.num_samples = args.num_samples
        self.buffer_size = args.buffer_size

    def add(self, data):
        indices = jnp.arange(self.idx, self.idx + self.num_samples) % self.buffer_size
        self.idx = self.idx + self.num_samples

        positions, forces, log_reward = data

        self.positions = self.positions.at[indices].set(positions)
        self.forces = self.forces.at[indices].set(forces)
        self.log_reward = self.log_reward.at[indices].set(log_reward)

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
    def __init__(self, args, sys):
        self.B = sys.B
        self.mass = sys.mass
        self.gamma = args.gamma
        self.sigma = args.sigma
        self.log_prob = sys.log_prob
        self.timestep = args.timestep

    def __call__(self, positions, forces):
        log_upm = self.unbiased_path_measure(positions, forces)
        log_ri = self.log_relaxed_indicator(positions[:, -1])

        log_reward = log_upm + log_ri
        return log_reward

    def unbiased_path_measure(self, positions, forces):
        velocities = (positions[:, 1:] - positions[:, :-1]) / self.timestep
        means = (1 - self.gamma * self.timestep) * velocities + forces[
            :, :-1
        ] * self.timestep / self.mass
        log_upm = self.log_prob(velocities[:, 1:] - means[:, :-1]).mean((1, 2))
        return log_upm

    def log_relaxed_indicator(self, position):
        pos = position.reshape(-1, position.shape[-1] // 3, 3)
        B = self.B.reshape(-1, position.shape[-1] // 3, 3)
        B = jnp.tile(B, (pos.shape[0], 1, 1))
        R, t = kabsch_align(pos, B)
        pos = jnp.matmul(pos, R.transpose(0, 2, 1)) + t
        pos = pos.reshape(*position.shape)
        log_rbf = -0.5 / self.sigma**2 * jnp.mean((pos - self.B) ** 2, axis=-1)
        return log_rbf
