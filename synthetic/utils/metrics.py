import jax.numpy as jnp


class Metric:
    def __init__(self, args, mds):
        self.device = args.device
        self.timestep = args.timestep
        self.save_dir = args.save_dir
        self.num_samples = args.num_samples

        self.U = mds.U
        self.std = mds.std
        self.dUdx = mds.dUdx
        self.log_prob = mds.log_prob
        self.target_position = mds.target_position

    def __call__(self, positions, potentials):
        final_position = positions[:, -1]
        rmsd, rmsd_std = self.rmsd(final_position, self.target_position)
        thp, hit = self.thp(final_position, self.target_position)
        etp, etp_std = self.etp(hit, potentials)

        metrics = {
            "rmsd": rmsd,
            "thp": 100 * thp,
            "etp": etp,
            "rmsd_std": rmsd_std,
            "etp_std": etp_std,
        }

        return metrics

    def rmsd(self, final_position, target_position):
        rmsd = jnp.sqrt(jnp.sum((final_position - target_position) ** 2, axis=-1))
        mean_rmsd, std_rmsd = jnp.mean(rmsd).item(), jnp.std(rmsd).item()
        return mean_rmsd, std_rmsd

    def thp(self, final_position, target_position):
        hit = jnp.sum((final_position - target_position) ** 2, axis=-1) < 0.5**2
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
