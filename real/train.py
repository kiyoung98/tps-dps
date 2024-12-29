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

parser.add_argument("--seed", default=0, type=int)

# Logging Config
parser.add_argument("--save_dir", default="results", type=str)
parser.add_argument("--save_freq", default=10, type=int)
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--project", default="aldp", type=str)


# System Config
parser.add_argument("--molecule", default="aldp", type=str)
parser.add_argument("--A_file", default="data/aldp/AD_A.pdb", type=str)
parser.add_argument("--B_file", default="data/aldp/AD_B.pdb", type=str)
parser.add_argument("--temperature", default=300, type=float)
parser.add_argument("--gamma", default=1.0, type=float)
parser.add_argument(
    "--forcefield",
    type=str,
    nargs="+",
    default=["amber14/protein.ff14SB.xml", "amber14/tip3p.xml"],
)

# Policy Config
parser.add_argument(
    "--bias", default="pot", type=str, choices=["pot", "force", "scale"]
)
parser.add_argument(
    "--hidden_dims", nargs="+", type=int, default=[256, 256, 256, 256, 256]
)

# Sampling Config
parser.add_argument("--sigma", default=0.1, type=float)
parser.add_argument("--num_steps", default=1000, type=int)
parser.add_argument("--timestep", default=1e-3, type=float)
parser.add_argument("--num_samples", default=16, type=int)

# Training Config
parser.add_argument("--start_temperature", default=600, type=float)
parser.add_argument("--end_temperature", default=300, type=float)
parser.add_argument("--num_rollouts", default=5000, type=int)
parser.add_argument("--policy_lr", default=1e-4, type=float)
parser.add_argument("--log_z_lr", default=1e-2, type=float)
parser.add_argument("--max_grad_norm", default=1, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--buffer_size", default=1000, type=int)
parser.add_argument("--trains_per_rollout", default=10, type=int)

args = parser.parse_args()

if __name__ == "__main__":
    for name in ["paths"]:
        os.makedirs(f"{args.save_dir}/{name}", exist_ok=True)
    if args.wandb:
        wandb.init(project=args.project, config=args)

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)

    sys = System(args)
    log = Log(args, sys)
    dps = DiffusionPathSampler(args, sys, init_key)

    kb = 1.380649 * 6.02214076 * 1e-3
    Ts = jnp.linspace(args.start_temperature, args.end_temperature, args.num_rollouts)
    kbTs = kb * Ts
    stds = [
        jnp.sqrt(2 * kbT * args.gamma / sys.mass) * jnp.sqrt(args.timestep)
        for kbT in kbTs
    ]

    print("Start training")
    for rollout in range(args.num_rollouts):
        print(f"Rollout: {rollout}")

        key, sample_key = jax.random.split(key)
        start = time.time()
        positions, potentials = dps.sample(args, sys, stds[rollout], sample_key)
        sampling_time = time.time() - start

        log.sample(dps, rollout, positions, potentials)

        start = time.time()
        key, train_key = jax.random.split(key)
        loss = dps.train(args, train_key)
        training_time = time.time() - start

        log.train(rollout, loss, sampling_time, training_time)
    print("End training")
