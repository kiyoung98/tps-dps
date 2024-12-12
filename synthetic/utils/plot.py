import jax.numpy as jnp
import matplotlib.pyplot as plt


class Plot:
    def __init__(self, args, mds):
        self.device = args.device
        self.save_dir = args.save_dir
        self.num_samples = args.num_samples

        self.U = mds.U
        self.start_position = mds.start_position
        self.target_position = mds.target_position
        self.xlim = mds.xlim
        self.ylim = mds.ylim

    def __call__(self, positions, rollout):
        self.paths(positions, rollout)

    def paths(self, positions, rollout):
        fig, ax = plt.subplots(figsize=(7, 7))

        zorder = 100

        plt.xlim(*self.xlim)
        plt.ylim(*self.ylim)
        x = jnp.linspace(*self.xlim, 400)
        y = jnp.linspace(*self.ylim, 400)
        X, Y = jnp.meshgrid(x, y)

        # X, Y to position input
        xs = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
        # calculate potential energy
        Z = self.U(xs).reshape(X.shape)

        ax.contourf(X, Y, Z, levels=30)

        cm = plt.get_cmap("gist_rainbow")

        ax.set_prop_cycle(
            color=[cm(1.0 * i / len(positions)) for i in range(len(positions))]
        )

        for position in positions:
            ax.plot(
                position[:, 0],
                position[:, 1],
                marker="o",
                linestyle="None",
                markersize=2,
                alpha=1.0,
                zorder=zorder - 1,
            )

        # Plot start and end positions
        ax.scatter(
            [self.start_position[0], self.target_position[0]],
            [self.start_position[1], self.target_position[1]],
            edgecolors="black",
            c="w",
            zorder=zorder,
        )

        # Plot basic configs
        ax.set_xlabel("x", fontsize=24, fontweight="medium")
        ax.set_ylabel("y", fontsize=24, fontweight="medium")
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
