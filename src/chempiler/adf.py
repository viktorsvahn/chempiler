"""Angular distribution function for intramolecular bond angles."""
import numpy as np
from ase.geometry import find_mic


def adf(frames, center="O", neighbors="H", formula=None,
        bins=90, angle_range=(80, 140)):
    """Distribution of (neighbors–center–neighbors) bond angles.

    For H-O-H angles in water::

        angles, density = adf(frames, "O", "H", formula="H2O")

    Parameters
    ----------
    frames : list of Frame
    center : str
        Element of the central atom (e.g. ``"O"``).
    neighbors : str
        Element of the flanking atoms (e.g. ``"H"``).
    formula : str, optional
        Restrict to molecules with this formula. ``None`` uses all molecules.
    bins : int
        Number of histogram bins.
    angle_range : tuple of (float, float)
        Angular range in degrees.

    Returns
    -------
    angles : numpy.ndarray
        Bin centres in degrees.
    density : numpy.ndarray
        Probability density (integrates to 1 over *angle_range*).
    """
    all_angles = []

    for frame in frames:
        pos = frame.atoms.get_positions()
        cell = frame.atoms.get_cell()
        syms = frame.symbols

        mol_ids = (
            frame.formula_to_mols.get(formula, [])
            if formula is not None
            else range(len(frame.molecules))
        )

        for mid in mol_ids:
            mol = frame.molecules[mid]
            c_atoms = [a for a in mol if syms[a] == center]
            n_atoms = [a for a in mol if syms[a] == neighbors]
            if len(n_atoms) < 2:
                continue

            n_pos = pos[np.array(n_atoms)]

            for c in c_atoms:
                vecs = n_pos - pos[c]
                vecs_mic, lens = find_mic(vecs, cell, pbc=True)
                vecs_unit = vecs_mic / (lens[:, None] + 1e-10)

                cos_mat = np.clip(vecs_unit @ vecs_unit.T, -1.0, 1.0)
                iu, ju = np.triu_indices(len(n_atoms), k=1)
                all_angles.extend(np.degrees(np.arccos(cos_mat[iu, ju])).tolist())

    if not all_angles:
        return np.array([]), np.array([])

    edges = np.linspace(angle_range[0], angle_range[1], bins + 1)
    counts, _ = np.histogram(all_angles, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    da = edges[1] - edges[0]
    total = counts.sum()
    density = counts / (total * da) if total > 0 else counts.astype(float)

    return centers, density
