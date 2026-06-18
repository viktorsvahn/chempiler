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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Jmol hue semantics (element recognisability) with Paul Tol's perceptually
# uniform, colourblind-safe values (https://personal.sron.nl/~pault/).
# Orange for P is borrowed from Tol vibrant — Tol bright has no orange.
_ELEM_COLOR = {
    'H':  '#BBBBBB',  # Tol grey   (Jmol: white — invisible on white bg)
    'C':  '#555555',  # dark grey  (Jmol: #909090 — darkened for contrast)
    'N':  '#4477AA',  # Tol blue
    'O':  '#EE6677',  # Tol red
    'F':  '#66CCEE',  # Tol cyan   (distinct from Cl green)
    'S':  '#CCBB44',  # Tol yellow (Jmol: bright yellow — weak on white bg)
    'Cl': '#228833',  # Tol green
    'P':  '#EE7733',  # Tol vibrant orange
}


def _parse_fspec(fspec):
    """Split 'path/to/file.xyz@3' into ('path/to/file.xyz', 3)."""
    if isinstance(fspec, (tuple, list)):
        return fspec[0], int(fspec[1])
    s = str(fspec)
    if '@' in s:
        path, idx = s.rsplit('@', 1)
        return path, int(idx)
    return s, 0


def _draw_cluster_2d(ax, atoms, margin=0.3, view=2):
    from ase.data import covalent_radii, atomic_numbers
    from matplotlib.patches import Circle
    pos = atoms.get_positions()
    syms = atoms.get_chemical_symbols()
    c = pos - pos.mean(axis=0)
    _, _, Vt = np.linalg.svd(c, full_matrices=True)
    plane = [i for i in range(3) if i != view]
    xy = c @ Vt[plane].T
    radii = np.array([covalent_radii[atomic_numbers[s]] for s in syms])

    for k, sym in enumerate(syms):
        r = radii[k]
        ax.add_patch(Circle(xy[k], radius=r,
                            color=_ELEM_COLOR.get(sym, '#BBBBBB'),
                            ec='#111', lw=0.65, zorder=2))
        ax.add_patch(Circle((xy[k][0] - 0.3 * r, xy[k][1] + 0.3 * r),
                            radius=0.18 * r,
                            color='white', alpha=0.6, lw=0, zorder=3))

    half = float(np.abs(xy).max() + radii.max())
    pad = half * margin
    ax.set_xlim(-half - pad, half + pad)
    ax.set_ylim(-half - pad, half + pad)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_rdf(r, g, insets=None, ax=None,
             inset_w=0.13, inset_h=0.38, inset_y=0.56, inset_margin=0.2,
             figsize=(10, 5), **line_kw):
    """Plot g(r) and optionally add per-peak structure insets.

    Parameters
    ----------
    r, g : array-like
        RDF arrays as returned by :func:`rdf`.
    insets : dict, optional
        ``{peak_r: fspec}`` mapping.  Each value is either:

        - a file path string ``"path.xyz"`` (reads frame 0)
        - ``"path.xyz@N"`` to select frame *N* from a multi-frame file
        - a dict ``{"file": fspec, "w": float, "h": float, "y": float,
          "color_scheme": str}`` to override per-inset options.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when *None*.
    inset_w, inset_h, inset_y : float
        Default inset width, height, and bottom edge in axes-fraction
        coordinates.  Per-inset dicts can override these individually.
    inset_margin : float
        Fractional padding added around the structure data in each inset.
    **line_kw
        Passed directly to ``ax.plot`` for the g(r) line.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from ase.io import read as ase_read

    r = np.asarray(r)
    g = np.asarray(g)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    line_kw.setdefault('lw', 1.5)
    ax.plot(r, g, **line_kw)
    ax.set_xlabel('r (Å)')
    ax.set_ylabel('g(r)')

    if insets:
        r_min, r_max = float(r[0]), float(r[-1])

        for rp, spec in insets.items():
            if isinstance(spec, dict):
                fspec = spec['file']
                scale = spec.get('scale', 1.0)
                w = spec.get('w', inset_w) * scale
                h = spec.get('h', inset_h) * scale
                y = spec.get('y', inset_y)
                m = spec.get('margin', inset_margin)
                view = spec.get('view', 2)
            else:
                fspec, w, h, y, m, view = spec, inset_w, inset_h, inset_y, inset_margin, 2

            path, idx = _parse_fspec(fspec)
            atoms = ase_read(path, index=idx)

            xf = (float(rp) - r_min) / (r_max - r_min)
            xf = float(np.clip(xf - w / 2, 0.01, 1.0 - w - 0.01))
            ins = ax.inset_axes([xf, y, w, h])
            _draw_cluster_2d(ins, atoms, margin=m, view=view)

    return ax
