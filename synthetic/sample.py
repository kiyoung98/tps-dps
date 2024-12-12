import os
import jax
import argparse
from system import System
from dps import DiffusionPathSampler

parser = argparse.ArgumentParser()

# System Config
parser.add_argument("--model_path", type=str)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--save_dir", default="paths/synthetic", type=str)

# Policy Config
parser.add_argument("--bias", default="pot", type=str)

# Sampling Config
parser.add_argument("--sigma", default=3, type=float)
parser.add_argument("--num_steps", default=1000, type=int)
parser.add_argument("--timestep", default=0.01, type=float)
parser.add_argument("--num_samples", default=1024, type=int)
parser.add_argument("--temperature", default=1200, type=float)

args = parser.parse_args()

if __name__ == "__main__":
    args.training = False
    for name in ["positions"]:
        if not os.path.exists(f"{args.save_dir}/{name}"):
            os.makedirs(f"{args.save_dir}/{name}")

    mds = System(args)
    agent = DiffusionPathSampler(args, mds)
    agent.policy.load_state_dict(jax.load(args.model_path))
    agent.sample(args, mds, mds.std)
