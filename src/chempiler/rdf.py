"""Radial distribution function (RDF) for reactive trajectories.

Pairs are resolved per-frame so that changing molecular populations (bond
breaking/formation) are handled correctly. Distances use ASE's find_mic for
minimum-image convention under periodic boundary conditions.
"""

import numpy as np
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list

from .selectors import resolve


def _default_rmax(frames):
    """Estimate a sensible rmax as half the mean cell width.

    Parameters
    ----------
    frames : list of Frame

    Returns
    -------
    float
        rmax in Ångström.
    """
    widths = [np.mean(np.linalg.norm(f.atoms.get_cell(), axis=1)) for f in frames]
    rmax = 0.5 * np.mean(widths)
    print(f"[RDF] rmax = {rmax:.3f} Å (auto)")
    return rmax


def rdf(frames, center, target, rmax=None, dr=0.02, integrate=False):
    """Compute the radial distribution function g(r).

    Parameters
    ----------
    frames : list of Frame
    center : str or dict
        Selector for the reference atoms (see selectors.resolve).
    target : str or dict
        Selector for the surrounding atoms.
    rmax : float, optional
        Maximum distance in Ångström. Defaults to half the mean cell width.
    dr : float
        Bin width in Ångström.
    integrate : bool
        If True, return the running coordination number n(r) instead of g(r).

    Returns
    -------
    r : numpy.ndarray
        Bin centres in Ångström.
    g : numpy.ndarray
        g(r) values, or n(r) if integrate is True.
    """
    if rmax is None:
        rmax = _default_rmax(frames)

    edges = np.arange(0.0, rmax + dr, dr)
    r = 0.5 * (edges[:-1] + edges[1:])
    shell = 4.0 * np.pi * r**2 * dr

    hist = np.zeros(len(r), dtype=np.float64)
    norm = 0.0
    rho_sum = 0.0
    n_contrib = 0
    nframes = len(frames)

    for frame in frames:
        pos = frame.atoms.get_positions()
        cell = frame.atoms.get_cell()

        c_idx = resolve(frame, center)
        t_idx = resolve(frame, target)

        if len(c_idx) == 0 or len(t_idx) == 0:
            continue

        volume = abs(np.linalg.det(cell))
        rho = len(t_idx) / volume
        norm += len(c_idx) * rho
        rho_sum += rho
        n_contrib += 1

        for i in c_idx:
            # Exclude i from t_idx to avoid a self-pair spike at r=0 when
            # center and target are the same species.
            sub_t = t_idx[t_idx != i]
            if len(sub_t) == 0:
                continue
            rij = pos[sub_t] - pos[i]
            rij, _ = find_mic(rij, cell, pbc=True)
            dist = np.linalg.norm(rij, axis=1)
            hist += np.histogram(dist, bins=edges)[0]

    norm /= nframes
    g = hist / (norm * shell * nframes)

    if integrate:
        # n(r) = rho_B * integral of g(r') 4pi r'^2 dr'
        rho_avg = rho_sum / n_contrib
        return r, rho_avg * np.cumsum(g * shell)
    return r, g
