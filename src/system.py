import jax
import jax.numpy as jnp
from openmm import app, unit
from dmff import Hamiltonian, NeighborList


class System:
    def __init__(self, args):
        U, self.A, self.B, self.mass = self.from_pdb(
            args.A_file, args.B_file, args.forcefield
        )
        kbT = 1.380649 * 6.02214076 * 1e-3 * args.temperature
        self.xi = jnp.sqrt(2 * kbT * args.gamma / self.mass)
        self.std = self.xi * jnp.sqrt(args.timestep)
        self.dim = self.A.shape[0]

        self.log_prob = lambda x: jax.scipy.stats.norm.logpdf(x, loc=0, scale=self.std)

        dUdx = jax.grad(lambda _x: U(_x).sum())

        self.U = jax.jit(U)
        self.dUdx = jax.jit(lambda _x: dUdx(_x))

    def from_pdb(self, A_file, B_file, forcefield):
        A_pdb, B_pdb = app.PDBFile(A_file), app.PDBFile(B_file)

        mass = [
            a.element.mass.value_in_unit(unit.dalton) for a in A_pdb.topology.atoms()
        ]
        mass = jnp.broadcast_to(jnp.array(mass).reshape(-1, 1), (len(mass), 3)).reshape(
            -1
        )

        A = jnp.array(A_pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        B = jnp.array(B_pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        num_atoms = A.shape[0]
        A, B = A.reshape(-1), B.reshape(-1)

        ff = Hamiltonian(*forcefield)
        potentials = ff.createPotential(
            A_pdb.topology,
            nonbondedMethod=app.NoCutoff,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=None,
            ewaldErrorTolerance=0.0005,
        )

        box = jnp.array([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 50.0]])
        nbList = NeighborList(box, 4.0, potentials.meta["cov_map"])
        nbList.allocate(A.reshape(-1, 3))

        _U = potentials.getPotentialFunc()

        @jax.vmap
        def U(_x):
            return _U(
                _x.reshape(num_atoms, 3), box, nbList.pairs, ff.paramset.parameters
            ).sum()

        return U, A, B, mass
