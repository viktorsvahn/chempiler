"""Tetrahedral order parameter q for four-coordinated networks."""
import numpy as np
from ase.geometry import find_mic


def tetrahedral_order(frames, element="O", n_neighbors=4, rcut=3.7):
    """Per-frame mean tetrahedral order parameter.

    q = 1 − (3/8) Σ_{j<k} (cos θ_jik + 1/3)²

    where j, k index the four nearest same-element neighbours of atom i
    and θ_jik is the angle at i between neighbours j and k.

    q = 1 for perfect tetrahedral coordination (ice); q ≈ 0.5–0.7 for
    liquid water; q ≈ 0 for an ideal gas.

    Parameters
    ----------
    frames : list of Frame
    element : str
        Central element symbol (``"O"`` for water).
    n_neighbors : int
        Number of nearest neighbours to include (4 for tetrahedral).
    rcut : float
        Distance cutoff in Å. Atoms with fewer than *n_neighbors* neighbours
        within this distance are excluded from the per-frame mean.

    Returns
    -------
    numpy.ndarray of shape (n_frames,)
        Mean q per frame. Frames with no qualifying atoms return ``nan``.
    """
    _iu, _ju = np.triu_indices(n_neighbors, k=1)
    q_series = []

    for frame in frames:
        pos = frame.atoms.get_positions()
        cell = frame.atoms.get_cell()
        syms = frame.symbols

        o_idx = np.array([i for i, s in enumerate(syms) if s == element])
        n_o = len(o_idx)
        if n_o < n_neighbors + 1:
            q_series.append(np.nan)
            continue

        o_pos = pos[o_idx]

        # All pairwise displacements in one find_mic call
        diffs = (o_pos[None, :, :] - o_pos[:, None, :]).reshape(-1, 3)
        vecs_mic, dists = find_mic(diffs, cell, pbc=True)
        vecs_mic = vecs_mic.reshape(n_o, n_o, 3)
        dists = dists.reshape(n_o, n_o)
        np.fill_diagonal(dists, np.inf)

        q_vals = []
        for i in range(n_o):
            near = np.argsort(dists[i])[:n_neighbors]
            if dists[i, near[-1]] > rcut:
                continue
            v = vecs_mic[i, near] / (dists[i, near, None] + 1e-10)
            cos_mat = np.clip(v @ v.T, -1.0, 1.0)
            q_vals.append(1.0 - (3.0 / 8.0) * np.sum((cos_mat[_iu, _ju] + 1.0 / 3.0) ** 2))

        q_series.append(float(np.mean(q_vals)) if q_vals else np.nan)

    return np.array(q_series)
