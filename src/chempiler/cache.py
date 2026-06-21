"""HDF5 cache for pre-built trajectory frames.

Serialises Frame objects (molecular topology + ASE Atoms) to an HDF5 file so
that the expensive perception step only runs once per trajectory. The cache is
keyed by a SHA-256 hash of the build parameters; a mismatch triggers a rebuild.

Molecules are stored in a flat CSR-style layout (flat_atoms + offsets) to avoid
variable-length HDF5 datasets. ASE Atoms objects are serialised as binary blobs
using ASE's own .traj format.
"""

import hashlib
import io
import json
import os

import h5py
import numpy as np
from ase.io import read, write

from .frame import Frame

CACHE_VERSION = 1


def make_cache_key(filename, mode, covalent_scale, coordination_scale,
                   max_frames, sphere_cutoffs=None, persistence=1):
    """Return a SHA-256 hex digest identifying a specific build configuration.

    Parameters
    ----------
    filename : str
        Absolute path to the trajectory file.
    mode : str
        Perception mode.
    covalent_scale : float
    coordination_scale : float
    max_frames : int or None
    sphere_cutoffs : dict or None
        {(elem_a, elem_b): float} used in sphere mode. Tuple keys are
        serialised as "ElemA-ElemB" strings so the payload is JSON-safe.

    Returns
    -------
    str
        64-character hex string.
    """
    payload = {
        "file": os.path.abspath(filename),
        "mode": mode,
        "covalent_scale": covalent_scale,
        "coordination_scale": coordination_scale,
        "max_frames": max_frames,
        "persistence": persistence,
        "version": CACHE_VERSION,
    }
    if sphere_cutoffs:
        payload["sphere_cutoffs"] = {
            f"{a}-{b}": v for (a, b), v in sorted(sphere_cutoffs.items())
        }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def save_cache(path, frames, key, mode, covalent_scale, coordination_scale):
    """Write frames to an HDF5 cache file.

    Parameters
    ----------
    path : str
        Output file path. Parent directories are created if needed.
    frames : list of Frame
    key : str
        Cache key returned by make_cache_key; stored as an attribute for
        validation on load.
    mode, covalent_scale, coordination_scale
        Build parameters stored as attributes (informational only).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with h5py.File(path, "w") as h5:
        h5.attrs["cache_key"] = key
        h5.attrs["version"] = CACHE_VERSION

        gframes = h5.create_group("frames")

        for i, f in enumerate(frames):
            g = gframes.create_group(str(i))

            g.create_dataset("formulas", data=np.array(f.formulas, dtype="S"))
            g.create_dataset("coms", data=np.asarray(f.coms), dtype=np.float64)

            # Molecules stored as a flat array + offset index (CSR layout).
            flat_atoms = []
            offsets = [0]
            for mol in f.molecules:
                flat_atoms.extend(mol)
                offsets.append(len(flat_atoms))

            g.create_dataset("flat_atoms", data=np.array(flat_atoms, dtype=np.int32))
            g.create_dataset("offsets", data=np.array(offsets, dtype=np.int32))

            # ASE Atoms serialised as a binary blob via the .traj format.
            buf = io.BytesIO()
            write(buf, f.atoms, format="traj")
            buf.seek(0)
            g.create_dataset("atoms", data=np.void(buf.read()))


def load_cache(path):
    """Load frames from an HDF5 cache file.

    Parameters
    ----------
    path : str
        Path to an existing cache file written by save_cache.

    Returns
    -------
    list of Frame
        Fully built Frame objects (build() has been called on each).

    Raises
    ------
    OSError
        If the file does not exist or cannot be opened.
    """
    frames = []

    with h5py.File(path, "r") as h5:
        for k in sorted(h5["frames"].keys(), key=int):
            g = h5["frames"][k]
            f = Frame()

            f.formulas = [
                x.decode() if isinstance(x, bytes) else str(x)
                for x in g["formulas"][:]
            ]
            f.coms = np.asarray(g["coms"])

            flat_atoms = g["flat_atoms"][:]
            offsets = g["offsets"][:]
            f.molecules = [
                list(flat_atoms[offsets[i]:offsets[i + 1]])
                for i in range(len(offsets) - 1)
            ]

            blob = g["atoms"][()].tobytes()
            f.atoms = read(io.BytesIO(blob), format="traj")

            # formulas and coms are already loaded; restore the remaining
            # derived fields without rerunning the per-molecule loop.
            f.symbols = f.atoms.get_chemical_symbols()
            f.positions = f.atoms.get_positions()
            for mi, formula in enumerate(f.formulas):
                f.formula_to_mols.setdefault(formula, []).append(mi)
            f._build_atom_to_mol()

            frames.append(f)

    return frames
