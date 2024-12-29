import wandb
import joblib
import mdtraj as md
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyemma.coordinates as coor

from .utils import *

class Plot:
    def __init__(self, args, sys):
        self.molecule = args.molecule
        self.save_dir = args.save_dir
        self.A_file = args.A_file
        self.B_file = args.B_file
        self.A = sys.A
        self.B = sys.B

    def __call__(self, positions, rollout):
        paths = self.paths(positions, rollout)

        plots = {
            "paths": wandb.Image(paths),
        }
        return plots

    def paths(self, positions, rollout):
        positions = positions.reshape(*positions.shape[:-1], -1, 3)
        zorder = 32
        circle_size = 1200

        plt.clf()
        plt.close()
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)

        cm = plt.get_cmap("gist_rainbow")
        ax.set_prop_cycle(
            color=[cm(1.0 * i / len(positions)) for i in range(len(positions))]
        )

        if self.molecule == "aldp":
            angle_2 = jnp.array([1, 6, 8, 14])
            angle_1 = jnp.array([6, 8, 14, 16])
            plt.xlim([-jnp.pi, jnp.pi])
            plt.ylim([-jnp.pi, jnp.pi])

            xs = jnp.arange(-jnp.pi, jnp.pi + 0.1, 0.1)
            ys = jnp.arange(-jnp.pi, jnp.pi + 0.1, 0.1)
            z = jnp.load(f"data/aldp/pmf.npy")

            plt.contourf(xs, ys, z, levels=100, zorder=0)

            for position in positions:
                psi = compute_torsion(position[:, angle_1])
                phi = compute_torsion(position[:, angle_2])

                ax.plot(
                    phi,
                    psi,
                    marker="o",
                    linestyle="None",
                    markersize=2,
                    alpha=1.0,
                )

            A_psi = compute_torsion(self.A.reshape(1, -1, 3)[:, angle_1])[0]
            A_phi = compute_torsion(self.A.reshape(1, -1, 3)[:, angle_2])[0]
            B_psi = compute_torsion(self.B.reshape(1, -1, 3)[:, angle_1])[0]
            B_phi = compute_torsion(self.B.reshape(1, -1, 3)[:, angle_2])[0]

            ax.scatter(
                [A_phi, B_phi],
                [A_psi, B_psi],
                edgecolors="black",
                c="w",
                zorder=zorder,
                s=circle_size,
            )

            plt.xlabel("\u03A6", fontsize=35, fontweight="medium")
            plt.ylabel("\u03A8", fontsize=35, fontweight="medium")
        else:
            xs = jnp.load(f"./data/{self.molecule}/xs.npy")
            ys = jnp.load(f"./data/{self.molecule}/ys.npy")
            plt.xlim(xs.min(), xs.max())
            plt.ylim(ys.min(), ys.max())

            z = jnp.load(f"./data/{self.molecule}/pmf.npy")
            plt.pcolormesh(xs, ys, z.T, cmap="viridis")

            tica_model = joblib.load(f"./data/{self.molecule}/tica_model.pkl")
            feat = coor.featurizer(self.A_file)
            feat.add_backbone_torsions(cossin=True)

            for position in positions:
                traj = md.Trajectory(
                    position,
                    md.load(self.A_file).topology,
                )
                feature = feat.transform(traj)
                tica = tica_model.transform(feature)
                ax.plot(
                    tica[:, 0],
                    tica[:, 1],
                    marker="o",
                    linestyle="None",
                    markersize=2,
                    alpha=1.0,
                )

            A = md.Trajectory(
                self.A.reshape(1, -1, 3),
                md.load(self.A_file).topology,
            )
            B = md.Trajectory(
                self.B.reshape(1, -1, 3),
                md.load(self.A_file).topology,
            )
            A_tica = tica_model.transform(feat.transform(A))[0]
            B_tica = tica_model.transform(feat.transform(B))[0]
            ax.scatter(
                [A_tica[0], B_tica[0]],
                [A_tica[1], B_tica[1]],
                edgecolors="black",
                c="w",
                zorder=zorder,
                s=circle_size,
            )
            plt.xlabel("TIC 1", fontsize=35, fontweight="medium")
            plt.ylabel("TIC 2", fontsize=35, fontweight="medium")

        plt.tick_params(
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False,
            bottom=False,
        )
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/paths/{rollout}.png")
        plt.show()
        plt.close()
        return fig
