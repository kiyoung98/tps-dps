import jax
import jax.numpy as jnp
from tqdm import tqdm
import optax
from bias import BiasForce


def loss_fn(params, positions, forces, log_tm, mds, args, policy):
    biases = policy.apply(params, positions, mds.target_position)
    means = positions + (forces + biases) * args.timestep
    log_bpm = mds.log_prob(positions[:, 1:] - means[:, :-1]).mean((1, 2))

    log_z = (log_tm - log_bpm).mean()
    loss = ((log_z + log_bpm - log_tm) ** 2).mean()
    return loss


class DiffusionPathSampler:
    def __init__(self, args, mds):
        self.policy = BiasForce()
        self.target_measure = TargetMeasure(args, mds)

        if args.training:
            positions = jnp.zeros(
                (args.batch_size, args.num_steps + 1, 2),
            )
            self.params = self.policy.init(args.key, positions, mds.target_position)
            self.optimizer = optax.adam(args.policy_lr)
            self.opt_state = self.optimizer.init(self.params)
            self.replay = ReplayBuffer(args)
            self.log_prob = mds.log_prob
            self.target_position = mds.target_position
            self.timestep = args.timestep
            self.grad_fn = jax.value_and_grad(loss_fn)

    def sample(self, args, mds, std, key, buffer=False):
        position = mds.start_position[None, :]
        for s in range(100):
            init_key, key = jax.random.split(key)
            position = (
                position
                - mds.dUdx(position) * args.timestep
                + mds.std * jax.random.normal(init_key, (args.num_samples, 2))
            )

        positions = jnp.zeros(
            (args.num_samples, args.num_steps + 1, 2),
        )
        forces = jnp.zeros(
            (args.num_samples, args.num_steps + 1, 2),
        )
        potentials = jnp.zeros(
            (args.num_samples, args.num_steps + 1),
        )
        noises = jax.random.normal(
            key,
            (args.num_samples, args.num_steps, 2),
        )

        force = -mds.dUdx(position)
        potential = mds.U(position)
        positions = positions.at[:, 0].set(position)
        forces = forces.at[:, 0].set(force)
        potentials = potentials.at[:, 0].set(potential)

        for s in tqdm(range(args.num_steps), desc="Sampling"):
            bias = self.policy.apply(
                self.params, position, mds.target_position
            ).squeeze()
            position = position + (force + bias) * args.timestep + std * noises[:, s]
            force = -mds.dUdx(position)

            positions = positions.at[:, s + 1].set(position)
            forces = forces.at[:, s + 1].set(force)
            potentials = potentials.at[:, s + 1].set(mds.U(position))

        log_tm = self.target_measure(positions, forces)

        if buffer:
            self.replay.add((positions, forces, log_tm))

        return positions, potentials

    def train(self, args, mds, key):
        loss_sum = 0
        for _ in tqdm(range(args.trains_per_rollout), desc="Training"):
            positions, forces, log_tm = self.replay.sample(key)
            loss, grads = self.grad_fn(
                self.params, positions, forces, log_tm, mds, args, self.policy
            )
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.params = optax.apply_updates(self.params, updates)

            loss_sum += loss

        loss = loss_sum / args.trains_per_rollout
        return loss


class ReplayBuffer:
    def __init__(self, args):
        self.positions = jnp.zeros(
            (args.buffer_size, args.num_steps + 1, 2),
        )
        self.forces = jnp.zeros(
            (args.buffer_size, args.num_steps + 1, 2),
        )
        self.log_reward = jnp.zeros(args.buffer_size)

        self.idx = 0
        self.key = args.key
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
        log_ri = self.log_relaxed_indicator(positions[:, -1], self.target_position)

        log_reward = log_upm + log_ri
        return log_reward

    def unbiased_path_measure(self, positions, forces):
        means = positions + forces * self.timestep
        log_upm = self.log_prob(positions[:, 1:] - means[:, :-1]).mean((1, 2))
        return log_upm

    def log_relaxed_indicator(self, final_position, target_position):
        log_rbf = (
            -0.5 / self.sigma**2 * ((final_position - target_position) ** 2).mean(-1)
        )
        return log_rbf
