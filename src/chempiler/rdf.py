"""Radial distribution function (RDF) for reactive trajectories.

Pairs are resolved per-frame so that changing molecular populations (bond
breaking/formation) are handled correctly. All (n_center × n_target)
displacement vectors are computed in a single numpy operation per frame and
passed to ASE's find_mic in one call, replacing the previous per-center-atom
Python loop.

Set n_workers > 1 to distribute frames across threads via
ThreadPoolExecutor. The inner loop is dominated by numpy (find_mic →
linalg.solve) which releases the GIL, so threads achieve real parallelism
without the serialization cost of multiprocessing.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import numpy as np
from ase.geometry import find_mic

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


def _rdf_worker(frames_chunk, center, target, edges):
    """Accumulate a partial RDF histogram over a chunk of frames.

    Module-level so it is picklable for ProcessPoolExecutor.

    Returns
    -------
    tuple of (hist, norm, rho_sum, n_contrib)
    """
    hist = np.zeros(len(edges) - 1, dtype=np.float64)
    norm = 0.0
    rho_sum = 0.0
    n_contrib = 0

    for frame in frames_chunk:
        c_idx = resolve(frame, center)
        t_idx = resolve(frame, target)

        if len(c_idx) == 0 or len(t_idx) == 0:
            continue

        pos = frame.atoms.get_positions()
        cell = frame.atoms.get_cell()

        volume = abs(np.linalg.det(cell))
        rho = len(t_idx) / volume
        norm += len(c_idx) * rho
        rho_sum += rho
        n_contrib += 1

        rij = (pos[t_idx][None, :, :] - pos[c_idx][:, None, :]).reshape(-1, 3)
        _, dists = find_mic(rij, cell, pbc=True)

        ci_grid = np.repeat(c_idx, len(t_idx))
        tj_grid = np.tile(t_idx, len(c_idx))
        hist += np.histogram(dists[ci_grid != tj_grid], bins=edges)[0]

    return hist, norm, rho_sum, n_contrib


def rdf(frames, center, target, rmax=None, dr=0.02, integrate=False, n_workers=1):
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
        If True, also compute and return the running coordination number n(r).
    n_workers : int
        Number of parallel threads. 1 (default) runs serially.
        Values > 1 split the frame list across threads using
        ThreadPoolExecutor. The dominant cost (numpy linalg) releases the
        GIL, so threads achieve real parallelism. n_workers=2 is a good
        starting point; returns diminish beyond the physical core count.

    Returns
    -------
    r : numpy.ndarray
        Bin centres in Ångström.
    g : numpy.ndarray
        g(r) values.
    n : numpy.ndarray
        Running coordination number n(r). Only returned when integrate is True.
    """
    if rmax is None:
        rmax = _default_rmax(frames)

    edges = np.arange(0.0, rmax + dr, dr)
    r = 0.5 * (edges[:-1] + edges[1:])
    shell = 4.0 * np.pi * r**2 * dr
    nframes = len(frames)

    if n_workers == 1:
        hist, norm, rho_sum, n_contrib = _rdf_worker(frames, center, target, edges)
    else:
        n_chunks = min(n_workers, nframes)
        chunk_size = math.ceil(nframes / n_chunks)
        chunks = [frames[i:i + chunk_size] for i in range(0, nframes, chunk_size)]

        hist = np.zeros(len(edges) - 1, dtype=np.float64)
        norm = 0.0
        rho_sum = 0.0
        n_contrib = 0

        with ThreadPoolExecutor(max_workers=n_chunks) as pool:
            for h, no, rs, nc in pool.map(
                _rdf_worker, chunks, repeat(center), repeat(target), repeat(edges)
            ):
                hist += h
                norm += no
                rho_sum += rs
                n_contrib += nc

    norm /= nframes
    if norm == 0.0:
        g = np.zeros_like(hist)
    else:
        g = hist / (norm * shell * nframes)

    if integrate:
        rho_avg = rho_sum / n_contrib
        n = rho_avg * np.cumsum(g * shell)
        return r, g, n
    return r, g
