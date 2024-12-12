import os
import jax
import jax.numpy as jnp
import argparse

from utils.logging import Log
from system import System
from dps import DiffusionPathSampler

parser = argparse.ArgumentParser()

# System Config
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--system", default="double_well_dual_channel", type=str)


# Logger Config
parser.add_argument("--save_dir", default="results/synthetic", type=str)

# Policy Config
parser.add_argument("--bias", default="pot", type=str)

# Sampling Config
parser.add_argument("--sigma", default=0.2, type=float)
parser.add_argument("--num_steps", default=1000, type=int)
parser.add_argument("--timestep", default=1e-3, type=float)
parser.add_argument("--num_samples", default=512, type=int)
# parser.add_argument("--temperature", default=1200, type=float)

# Training Config
# parser.add_argument("--start_temperature", default=2400, type=float)
# parser.add_argument("--end_temperature", default=1200, type=float)
parser.add_argument("--num_rollouts", default=20, type=int)
parser.add_argument("--max_grad_norm", default=1, type=int)
parser.add_argument("--xi", default=1e-1, type=float)
parser.add_argument("--start_xi", default=1, type=float)
parser.add_argument("--end_xi", default=1, type=float)
# parser.add_argument("--log_z_lr", default=1e-3, type=float)
parser.add_argument(
    "--policy_lr",
    default=5e-4,
    type=float,
)
parser.add_argument("--batch_size", default=4048, type=int)
parser.add_argument(
    "--buffer_size",
    default=10000,
    type=int,
)
parser.add_argument(
    "--trains_per_rollout",
    default=100,
    type=int,
)

args = parser.parse_args()

if __name__ == "__main__":
    args.training = True
    for name in ["policies", "positions", "paths"]:
        if not os.path.exists(f"{args.save_dir}/{name}"):
            os.makedirs(f"{args.save_dir}/{name}")

    args.key = jax.random.PRNGKey(args.seed)
    key = args.key

    # check if we can use gpu
    if jax.device_count() == 0:
        print("No GPU found. Using CPU")

    mds = System(args)
    log = Log(args, mds)
    agent = DiffusionPathSampler(args, mds)

    # temperatures = jnp.linspace(
    #     args.start_temperature, args.end_temperature, args.num_rollouts
    # )

    # stds = jnp.sqrt(2 * mds.kB * args.timestep * temperatures)
    # std = jnp.sqrt(2 * mds.kB * args.timestep * args.temperature)

    std = args.xi * jnp.sqrt(args.timestep)
    stds = jnp.linspace(args.start_xi, args.end_xi, args.num_rollouts) * jnp.sqrt(
        args.timestep
    )

    log.info("Start training")
    for rollout in range(args.num_rollouts):
        print(f"Rollout: {rollout}")

        key, _ = jax.random.split(key)
        agent.sample(args, mds, stds[rollout], key, True)
        loss = agent.train(args, mds, key)
        log.logger.info(f"loss: {loss}")
        positions, potentials = agent.sample(args, mds, std, key, False)
        log.sample(rollout, agent.policy, positions, potentials)
    log.info("End training")

    key, _ = jax.random.split(key)
    positions, potentials = agent.sample(args, mds, std, key)
    log.sample(args.num_rollouts, agent.policy, positions, potentials)
