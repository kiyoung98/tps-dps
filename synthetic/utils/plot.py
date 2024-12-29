import wandb
import seaborn as sns
import jax.numpy as jnp
import matplotlib.pyplot as plt


class Plot:
    def __init__(self, args, mds):
        self.device = args.device
        self.save_dir = args.save_dir
        self.num_samples = args.num_samples

        self.U = mds.U
        self.dUdx = mds.dUdx
        self.xlim = mds.xlim
        self.ylim = mds.ylim
        self.start_position = mds.start_position
        self.target_position = mds.target_position

    def __call__(self, positions, potentials, agent, rollout):
        paths = self.paths(positions, rollout)
        force_field = self.force_field(agent, rollout)
        energy_distribution = self.energy_distribution(potentials, rollout)
        y_distribution = self.y_distribution(positions, potentials, rollout)

        plots = {
            "paths": wandb.Image(paths),
            "force_field": wandb.Image(force_field),
            "y_distribution": wandb.Image(y_distribution),
            "energy_distribution": wandb.Image(energy_distribution),
        }
        return plots

    def force_field(self, agent, rollout):
        fig, ax = plt.subplots(figsize=(7, 7))
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

        ax.set_xlabel("x", fontsize=40, fontweight="medium")
        ax.set_ylabel("y", fontsize=40, fontweight="medium")
        ax.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
        plt.tight_layout()

        # get force field for 20x20 grid on landscape
        fx = jnp.linspace(*self.xlim, 10)
        fy = jnp.linspace(*self.ylim, 10)
        fX, fY = jnp.meshgrid(fx, fy)

        pos = jnp.stack([fX, fY], axis=-1)

        force = -self.dUdx(pos.reshape(-1, 2)).reshape(pos.shape)
        bias_force = agent.policy.apply(agent.params, pos, self.target_position)

        total_force = force + bias_force
        ax.quiver(fX, fY, total_force[:, :, 0], total_force[:, :, 1])
        plt.savefig(f"{self.save_dir}/force_fields/{rollout}.png")

        # [X, Y] to jax tensor
        pos = jnp.stack([X, Y], axis=-1)
        plt.close()
        return fig

    def energy_distribution(self, potentials, rollout):
        # continuously 1D plot the energy distribution of the transition state
        transition_state_energy = []
        for i in range(len(potentials)):
            transition_state_energy.append(potentials[i].max().item())

        # plot the energy probability distribution
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.kdeplot(transition_state_energy, color="red", fill=True, alpha=0.5, ax=ax)

        ax.set_xlabel("Potential Energy (kJ/mol)", fontsize=24, fontweight="medium")
        ax.set_ylabel("")
        # Plot basic configs
        plt.tick_params(
            left=False,
            right=False,
            labelleft=False,
            labelbottom=True,
            bottom=True,
            labelsize=14,
        )
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/energy_distributions/{rollout}.png")
        plt.show()
        plt.close()
        return fig

    def y_distribution(self, positions, potentials, rollout):
        ys = []
        for i in range(len(positions)):
            idx = potentials[i].argmax()
            ys.append(positions[i][idx][1].item())

        fig, ax = plt.subplots(figsize=(7, 7))
        sns.kdeplot(ys, color="red", fill=True, alpha=0.5, ax=ax)

        ax.set_xlabel("y", fontsize=24, fontweight="medium")
        ax.set_ylabel("")
        # Plot basic configs
        plt.tick_params(
            left=False,
            right=False,
            labelleft=False,
            labelbottom=True,
            bottom=True,
            labelsize=14,
        )
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/y_distributions/{rollout}.png")
        plt.show()
        plt.close()
        return fig

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
