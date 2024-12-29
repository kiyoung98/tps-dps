from .utils import *
import jax.numpy as jnp


class Metric:
    def __init__(self, args, sys):
        self.A_file = args.A_file
        self.B_file = args.B_file
        self.molecule = args.molecule

        self.U = sys.U
        self.B = sys.B

    def __call__(self, positions, potentials):
        positions = positions.reshape(*positions.shape[:-1], -1, 3)
        final_position = positions[:, -1]
        rmsd, rmsd_std = self.rmsd(final_position)
        thp, hit = self.thp(final_position)
        etp, etp_std = self.etp(hit, potentials)

        metrics = {
            "rmsd": rmsd,
            "thp": 100 * thp,
            "etp": etp,
            "rmsd_std": rmsd_std,
            "etp_std": etp_std,
        }

        return metrics

    def rmsd(self, position):
        B = self.B.reshape(-1, self.B.shape[-1] // 3, 3)
        B = jnp.tile(B, (position.shape[0], 1, 1))
        R, t = kabsch_align(position, B)
        position = jnp.matmul(position, R.transpose(0, 2, 1)) + t
        rmsd = jnp.sqrt(jnp.sum((position - B) ** 2, axis=-1).mean(-1))
        mean_rmsd, std_rmsd = jnp.mean(rmsd).item(), jnp.std(rmsd).item()
        return mean_rmsd, std_rmsd

    def thp(self, position):
        B = self.B.reshape(1, -1, 3)
        if self.molecule == "aldp":
            cv1_diff, cv2_diff = alanine_torsion_diff(position, B)
        else:
            cv1_diff, cv2_diff = tic_diff(self.molecule, self.A_file, self.B_file, position, B)
        hit = (cv1_diff**2 + cv2_diff**2) < 0.75**2
        thp = jnp.sum(hit).astype(float) / len(hit)
        return thp.item(), hit

    def etp(self, hit, potentials):
        etps = []
        for i, hit_idx in enumerate(hit):
            if hit_idx:
                etp = jnp.max(potentials[i], axis=0)
                etps.append(etp)

        if len(etps) > 0:
            etps = jnp.array(etps)
            etp, std_etp = jnp.mean(etps).item(), jnp.std(etps).item()
            return etp, std_etp
        else:
            return None, None