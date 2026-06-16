"""Sphere-mode molecular perception for coordination chemistry.

Instead of a global covalent bond graph, sphere mode:
1. Builds tight covalent molecules for all non-centre atoms (H2O, Cl-, etc.)
   using skin=0 to exclude hydrogen-bond distances from registering as bonds.
2. Groups those base molecules around coordination centres (Be, Na, ...)
   using explicit distance cutoffs placed at the first minimum of the
   centre-ligand RDF — the natural density gap between first and second shell.

Cutoffs can be supplied by the user ({'Be-O': 2.4}) or auto-detected by
sampling a subset of trajectory frames.
"""

import numpy as np
from ase.geometry import find_mic
from ase.neighborlist import NeighborList, natural_cutoffs

from .perception import UnionFind

# Elements that form standard covalent molecules; everything else is treated
# as a potential coordination centre when auto-detecting.
_COVALENT_ELEMENTS = frozenset([
    'H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl',
    'Ge', 'As', 'Se', 'Br', 'I',
])

_DEFAULT_LIGANDS = ('O', 'N', 'S', 'Cl', 'F')


def parse_cutoffs(raw):
    """Normalise a user-supplied cutoff mapping to {(elem_a, elem_b): float}.

    Accepts both string keys (``'Be-O'``) and tuple keys (``('Be', 'O')``).

    Parameters
    ----------
    raw : dict or None

    Returns
    -------
    dict
        {(str, str): float}
    """
    if not raw:
        return {}
    out = {}
    for k, v in raw.items():
        if isinstance(k, str):
            parts = k.split('-')
            if len(parts) != 2:
                raise ValueError(f"Cutoff key {k!r} must be 'ElemA-ElemB'.")
            a, b = parts[0].strip(), parts[1].strip()
        else:
            a, b = str(k[0]), str(k[1])
        out[(a, b)] = float(v)
    return out


def detect_coordination_centers(atoms_list):
    """Return element symbols present in the sample that are not common covalent elements.

    Parameters
    ----------
    atoms_list : list of ase.Atoms

    Returns
    -------
    set of str
    """
    present = set()
    for atoms in atoms_list:
        present.update(atoms.get_chemical_symbols())
    return present - _COVALENT_ELEMENTS


def detect_sphere_cutoffs(atoms_list, centers,
                          ligands=_DEFAULT_LIGANDS, rmax=5.0, dr=0.05):
    """Auto-detect sphere cutoffs from pair distance histograms.

    For each (centre, ligand) pair, a distance histogram is built over the
    sample frames, smoothed, and then searched for the first minimum after the
    first peak. That minimum marks the density gap between the first and second
    coordination shell and is used as the sphere cutoff.

    Parameters
    ----------
    atoms_list : list of ase.Atoms
        Sample frames (typically 50–100).
    centers : iterable of str
        Element symbols to treat as coordination centres.
    ligands : tuple of str
        Ligand elements to pair with each centre.
    rmax : float
        Maximum distance in Å.
    dr : float
        Histogram bin width in Å.

    Returns
    -------
    dict
        {(elem_a, elem_b): cutoff_Å}
    """
    edges = np.arange(0.0, rmax + dr, dr)
    r = 0.5 * (edges[:-1] + edges[1:])
    cutoffs = {}

    for elem_a in sorted(centers):
        for elem_b in ligands:
            hist = np.zeros(len(r), dtype=np.float64)
            n_frames = 0

            for atoms in atoms_list:
                syms = atoms.get_chemical_symbols()
                if elem_a not in syms or elem_b not in syms:
                    continue

                pos  = atoms.get_positions()
                cell = atoms.get_cell()
                pbc  = atoms.get_pbc()

                idx_a = np.array([i for i, s in enumerate(syms) if s == elem_a])
                idx_b = np.array([i for i, s in enumerate(syms) if s == elem_b])

                diff = (pos[idx_b][None, :, :] - pos[idx_a][:, None, :]).reshape(-1, 3)
                _, dists = find_mic(diff, cell, pbc=pbc)
                dists = dists[dists > 0.3]  # exclude self-pairs / covalent bonds
                hist += np.histogram(dists, bins=edges)[0]
                n_frames += 1

            if n_frames == 0 or hist.sum() == 0:
                continue

            # Smooth with a ~0.25 Å window to suppress statistical noise
            window = max(1, int(round(0.25 / dr)))
            g = np.convolve(hist, np.ones(window) / window, mode='same')

            # Locate the first significant peak within the first half of rmax
            search_end = max(2, int(0.55 * rmax / dr))
            peak_i = int(np.argmax(g[1:search_end])) + 1
            if g[peak_i] < 1.0:
                continue  # no real peak found

            # First minimum after that peak
            for i in range(peak_i + 1, len(g) - 1):
                if g[i] <= g[i - 1] and g[i] <= g[i + 1]:
                    cutoffs[(elem_a, elem_b)] = float(r[i])
                    print(
                        f"[Chempiler] sphere cutoff {elem_a}–{elem_b}: "
                        f"{r[i]:.2f} Å (auto, first peak at {r[peak_i]:.2f} Å)"
                    )
                    break

    return cutoffs


def build_molecules_sphere(atoms, sphere_cutoffs, bond_scale=1.0):
    """Build molecular groups via tight covalent bonds and sphere cutoffs.

    Centre atoms are excluded from the covalent bond step so they always start
    as isolated single-atom groups. A NeighborList with ``skin=0`` is used so
    that hydrogen-bond O···H distances (~1.5–2.5 Å) never register as O–H
    covalent bonds. Each centre atom then absorbs all base-molecule groups
    whose ligand atom falls within the specified sphere cutoff.

    Parameters
    ----------
    atoms : ase.Atoms
    sphere_cutoffs : dict
        {(elem_a, elem_b): cutoff_Å} as returned by :func:`parse_cutoffs`
        or :func:`detect_sphere_cutoffs`.
    bond_scale : float
        Scale factor applied to ASE natural cutoffs in the covalent step.

    Returns
    -------
    list of list of int
    """
    syms = atoms.get_chemical_symbols()
    n    = len(atoms)
    pos  = atoms.get_positions()
    cell = atoms.get_cell()
    pbc  = atoms.get_pbc()

    center_elements = {a for (a, _) in sphere_cutoffs}

    # ── Step 1: tight covalent graph, centre atoms excluded from bonding ──────
    cov = natural_cutoffs(atoms)
    for i, s in enumerate(syms):
        if s in center_elements:
            cov[i] = 0.0
        else:
            cov[i] *= bond_scale

    # skin=0.1 Å: enough margin to capture slightly-stretched covalent bonds
    # (~0.97 Å O-H → cutoff 1.07 Å) while keeping H-bonds (>1.5 Å) excluded.
    nl = NeighborList(cov, skin=0.1, self_interaction=False, bothways=True)
    nl.update(atoms)

    uf = UnionFind(n)
    for i in range(n):
        for j in nl.get_neighbors(i)[0]:
            uf.union(i, j)

    # ── Step 2: merge each centre atom with nearby ligand groups ──────────────
    for i, s_i in enumerate(syms):
        if s_i not in center_elements:
            continue
        for (elem_a, elem_b), cutoff in sphere_cutoffs.items():
            if s_i != elem_a:
                continue
            j_arr = np.array(
                [j for j, s_j in enumerate(syms) if s_j == elem_b],
                dtype=np.int32,
            )
            if j_arr.size == 0:
                continue
            diff = pos[j_arr] - pos[i]
            _, dists = find_mic(diff, cell, pbc=pbc)
            for k in np.where(dists < cutoff)[0]:
                uf.union(i, int(j_arr[k]))

    # ── Collect groups ────────────────────────────────────────────────────────
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    return list(groups.values())
