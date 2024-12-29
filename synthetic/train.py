import os
import jax
import time
import wandb
import argparse
import jax.numpy as jnp
from system import System
from utils.logging import Log
from dps import DiffusionPathSampler

parser = argparse.ArgumentParser()

# System Config
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--project", default="synthetic", type=str)
parser.add_argument("--system", default="double_well_dual_channel", type=str)


# Logger Config
parser.add_argument("--save_dir", default="results/synthetic", type=str)

# Policy Config
parser.add_argument("--bias", default="pot", type=str)
parser.add_argument("--hidden_dim", default=32, type=int)

# Sampling Config
parser.add_argument("--sigma", default=0.4, type=float)
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
parser.add_argument("--start_xi", default=0.5, type=float)
parser.add_argument("--end_xi", default=0.5, type=float)
# parser.add_argument("--log_z_lr", default=1e-3, type=float)
parser.add_argument(
    "--policy_lr",
    default=1e-3,
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
    key = jax.random.PRNGKey(args.seed)
    for name in [
        "policies",
        "positions",
        "paths",
        "force_fields",
        "energy_distributions",
        "y_distributions",
    ]:
        if not os.path.exists(f"{args.save_dir}/{name}"):
            os.makedirs(f"{args.save_dir}/{name}")
    if args.wandb:
        wandb.init(project=args.project, config=args)

    mds = System(args)
    log = Log(args, mds)
    dps = DiffusionPathSampler(args, mds, key)

    std = args.xi * jnp.sqrt(args.timestep)
    stds = jnp.linspace(args.start_xi, args.end_xi, args.num_rollouts) * jnp.sqrt(
        args.timestep
    )

    log.info("Start training")
    for rollout in range(args.num_rollouts):
        print(f"Rollout: {rollout}")
        start = time.time()
        key, subkey = jax.random.split(key)
        positions, potentials = dps.sample(args, mds, stds[rollout], key)
        log.sample(rollout, dps, positions, potentials)
        loss = dps.train(args, subkey)
        print(f"Time: {time.time() - start}")
        log.logger.info(f"loss: {loss}")
    log.info("End training")

    key = jax.random.split(key)[0]
    positions, potentials = dps.sample(args, mds, std, key)
    log.sample(args.num_rollouts, dps, positions, potentials)
