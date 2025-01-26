import jax
import joblib
import mdtraj as md
import jax.numpy as jnp
import pyemma.coordinates as coor

@jax.jit
@jax.vmap
def kabsch_align(P, Q):
    centroid_P = jnp.mean(P, axis=0, keepdims=True)
    centroid_Q = jnp.mean(Q, axis=0, keepdims=True)

    p = P - centroid_P
    q = Q - centroid_Q

    H = jnp.matmul(p.T, q)

    U, S, Vt = jnp.linalg.svd(H)

    d = jnp.linalg.det(jnp.matmul(Vt.T, U.T))

    Vt = jnp.where(d < 0.0, Vt.at[-1, :].set(Vt[-1, :] * -1.0), Vt)

    R = jnp.matmul(Vt.T, U.T)
    t = centroid_Q - jnp.matmul(centroid_P, R.T)
    return R, t


@jax.jit
def compute_torsion(positions):
    v = positions[:, :-1] - positions[:, 1:]
    v0 = -v[:, 0]
    v1 = v[:, 2]
    v2 = v[:, 1]

    s0 = jnp.sum(v0 * v2, axis=-1, keepdims=True) / jnp.sum(
        v2 * v2, axis=-1, keepdims=True
    )
    s1 = jnp.sum(v1 * v2, axis=-1, keepdims=True) / jnp.sum(
        v2 * v2, axis=-1, keepdims=True
    )

    v0 = v0 - s0 * v2
    v1 = v1 - s1 * v2

    v0 = v0 / jnp.linalg.norm(v0, axis=-1, keepdims=True)
    v1 = v1 / jnp.linalg.norm(v1, axis=-1, keepdims=True)
    v2 = v2 / jnp.linalg.norm(v2, axis=-1, keepdims=True)

    x = jnp.sum(v0 * v1, axis=-1)
    v3 = jnp.cross(v0, v2, axis=-1)
    y = jnp.sum(v3 * v1, axis=-1)
    return jnp.arctan2(y, x)

def tic_diff(molecule, A_file, B_file, position, B):
    tica_model = joblib.load(f"./data/{molecule}/tica_model.pkl")
    feat = coor.featurizer(A_file)
    feat.add_backbone_torsions(cossin=True)
    traj = md.Trajectory(
        position,
        md.load(A_file).topology,
    )
    feature = feat.transform(traj)
    tica = tica_model.transform(feature)

    traj = md.Trajectory(
        B,
        md.load(B_file).topology,
    )
    feature = feat.transform(traj)
    B_tica = tica_model.transform(feature)

    tic1_diff = jnp.abs(tica[:, 0] - B_tica[:, 0])
    tic2_diff = jnp.abs(tica[:, 1] - B_tica[:, 1])
    return tic1_diff, tic2_diff

def aldp_torsion_diff(position, B):
    angle_2 = jnp.array([1, 6, 8, 14])
    angle_1 = jnp.array([6, 8, 14, 16])

    B_psi = compute_torsion(B[:, angle_1])
    B_phi = compute_torsion(B[:, angle_2])

    psi = compute_torsion(position[:, angle_1])
    phi = compute_torsion(position[:, angle_2])

    psi_diff = jnp.abs(psi - B_psi) % (2 * jnp.pi)
    psi_diff = jnp.minimum(psi_diff, 2 * jnp.pi - psi_diff)

    phi_diff = jnp.abs(phi - B_phi) % (2 * jnp.pi)
    phi_diff = jnp.minimum(phi_diff, 2 * jnp.pi - phi_diff)
    return psi_diff, phi_diff
