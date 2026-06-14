"""State functions for use with Tracker.

A state function maps (frame, entity) → hashable state. The functions here
encode chemical state assignments based on spatial proximity.
"""

import numpy as np


def nearest_host(frame, atom_idx, host_symbol, cutoff):
    """Return the index of the nearest host atom within cutoff, or None.

    Uses the minimum image convention so bonds that straddle a periodic
    boundary are handled correctly.

    Parameters
    ----------
    frame : Frame
    atom_idx : int
        Index of the atom whose host assignment is being determined.
    host_symbol : str
        Element symbol of the host species (e.g. ``"O"``).
    cutoff : float
        Maximum bond distance in Ångström.

    Returns
    -------
    int or None
        Atom index of the nearest host within *cutoff*, or None if the atom
        is unbound (no host within cutoff).
    """
    pos = frame.positions
    host_idx = np.array(
        [i for i, s in enumerate(frame.symbols) if s == host_symbol and i != atom_idx],
        dtype=np.int32,
    )
    if len(host_idx) == 0:
        return None

    diffs = pos[host_idx] - pos[atom_idx]

    if frame.atoms.get_pbc().any():
        cell = np.asarray(frame.atoms.get_cell())
        # Minimum image correction: subtract nearest lattice vector.
        diffs -= np.round(diffs @ np.linalg.inv(cell)) @ cell

    dists = np.linalg.norm(diffs, axis=1)
    within = dists < cutoff

    if not within.any():
        return None

    return int(host_idx[within][np.argmin(dists[within])])
