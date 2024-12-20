import torch
import numpy as np
from .utils import *


class Metric:
    def __init__(self, args, mds):
        self.device = args.device
        self.molecule = args.molecule
        self.save_dir = args.save_dir
        self.timestep = args.timestep
        self.friction = args.friction
        self.num_samples = args.num_samples

        self.m = mds.m
        self.std = mds.std
        self.log_prob = mds.log_prob
        self.heavy_atoms = mds.heavy_atoms
        self.energy_function = mds.energy_function
        self.target_position = mds.target_position

    def __call__(self):
        positions, forces, potentials = [], [], []
        for i in range(self.num_samples):
            position = np.load(f"{self.save_dir}/positions/{i}.npy").astype(np.float32)
            force, potential = self.energy_function(position)
            positions.append(torch.from_numpy(position).to(self.device))
            forces.append(torch.from_numpy(force).to(self.device))
            potentials.append(torch.from_numpy(potential).to(self.device))

        final_position = torch.stack([position[-1] for position in positions])
        rmsd, rmsd_std = self.rmsd(
            final_position[:, self.heavy_atoms],
            self.target_position[:, self.heavy_atoms],
        )
        thp, hit = self.thp(final_position, self.target_position)
        etp, etp_std = self.etp(hit, potentials)
        metrics = {
            "rmsd": 10 * rmsd,
            "thp": 100 * thp,
            "etp": etp,
            "rmsd_std": 10 * rmsd_std,
            "etp_std": etp_std,
        }
        return metrics

    def rmsd(self, position, target_position):
        R, t = kabsch(position, target_position)
        position = torch.matmul(position, R.transpose(-2, -1)) + t
        rmsd = (position - target_position).square().sum(-1).mean(-1).sqrt()
        rmsd, std_rmsd = rmsd.mean().item(), rmsd.std().item()
        return rmsd, std_rmsd

    def thp(self, position, target_position):
        if self.molecule == "alanine":
            psi_diff, phi_diff = alanine_diff(position, target_position)
            hit = psi_diff.square() + phi_diff.square() < 0.75**2

        elif self.molecule == "poly":
            handed = poly_handed(position)
            hit = handed > 0

        else:
            tic1_diff, tic2_diff = tic_diff(self.molecule, position, target_position)
            hit = tic1_diff.square() + tic2_diff.square() < 0.75**2

        hit = hit.squeeze()
        thp = hit.sum().float() / len(hit)

        return thp.item(), hit

    def etp(self, hit, potentials):
        etps = []
        for i, hit_idx in enumerate(hit):
            if hit_idx:
                etp = potentials[i].max(0)[0]
                etps.append(etp)

        if len(etps) > 0:
            etps = torch.tensor(etps)
            etp, std_etp = etps.mean().item(), etps.std().item()
            return etp, std_etp
        else:
            return None, None
