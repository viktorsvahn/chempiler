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
from ase.io import read as ase_read

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
# Shell analysis
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class ShellInfo:
    """Coordination shell structure extracted from a g(r) curve.

    Attributes
    ----------
    peaks : np.ndarray
        Detected peak positions (Å).
    troughs : np.ndarray
        Shell boundaries (Å), length ``len(peaks) + 1``.  Always starts at
        0.0 and ends at the last r value of the input grid.
    cns : np.ndarray or None
        Cumulative coordination number at the outer boundary of each shell,
        i.e. ``cns[i] = n(troughs[i+1])``, length ``len(peaks)``.
        ``None`` when no coordination array was supplied.
        Per-shell (annular) CNs are ``np.diff(cns)`` if needed.
    """
    peaks: np.ndarray
    troughs: np.ndarray
    cns: "np.ndarray | None"


def rdf_shells(r, g, n=None, *, peak_min_g=0.5, smooth_sigma=0.0):
    """Detect coordination shells in a pre-computed g(r).

    Parameters
    ----------
    r, g : array-like
        Distance grid and g(r) values as returned by :func:`rdf`.
    n : array-like, optional
        Running coordination number from :func:`rdf` (``integrate=True``).
        When supplied, per-shell and cumulative CNs are computed.
    peak_min_g : float
        Minimum g(r) height to qualify as a peak (default 0.5).
    smooth_sigma : float
        Standard deviation in Å of a Gaussian applied to g(r) before
        peak/trough detection.  ``0`` disables smoothing (default).

    Returns
    -------
    ShellInfo
        ``.peaks`` — peak positions; ``.troughs`` — shell boundaries
        (length ``len(peaks) + 1``, starts at 0.0); ``.cns[i]`` —
        cumulative CN at ``troughs[i+1]``.  ``None`` when *n* is not
        supplied.  Per-shell annular CNs are ``np.diff(cns)``.
    """
    r = np.asarray(r, dtype=float)
    g = np.asarray(g, dtype=float)

    if smooth_sigma > 0.0:
        dr_step = float(r[1] - r[0])
        sigma_bins = smooth_sigma / dr_step
        half_w = max(1, int(4 * sigma_bins))
        x = np.arange(-half_w, half_w + 1, dtype=float)
        kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
        kernel /= kernel.sum()
        g_det = np.convolve(g, kernel, mode='same')
    else:
        g_det = g

    valid = r > 0.5
    r_v, g_v = r[valid], g_det[valid]
    is_peak = (
        (g_v[1:-1] > g_v[:-2]) &
        (g_v[1:-1] > g_v[2:]) &
        (g_v[1:-1] > peak_min_g)
    )
    peaks = r_v[1:-1][is_peak].copy()

    troughs = [0.0]
    for i in range(len(peaks) - 1):
        r1, r2 = float(peaks[i]), float(peaks[i + 1])
        mask = (r >= r1) & (r <= r2)
        if mask.sum() < 3:
            troughs.append(0.5 * (r1 + r2))
        else:
            troughs.append(float(r[mask][int(np.argmin(g_det[mask]))]))
    troughs.append(float(r[-1]))
    troughs = np.array(troughs)

    if n is not None:
        n_arr = np.asarray(n, dtype=float)
        cns = np.array([float(np.interp(t, r, n_arr)) for t in troughs[1:]])
    else:
        cns = None

    return ShellInfo(peaks=peaks, troughs=troughs, cns=cns)


class ShellEnvironments(dict):
    """dict of ``{peak_r: [ase.Atoms]}`` with an attached :class:`ShellInfo`.

    Behaves exactly like a plain dict (drop-in for *plot_rdf* ``insets``),
    but also exposes ``.shells`` so peaks, troughs, and CNs are accessible
    without recomputing them.
    """

    def __init__(self, data, shells: ShellInfo):
        super().__init__(data)
        self.shells = shells


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


def _draw_cluster_2d(ax, atoms, margin=0.3, view=2, view_range=None):
    from ase.data import covalent_radii, atomic_numbers
    from matplotlib.patches import Circle
    pos = atoms.get_positions()
    syms = atoms.get_chemical_symbols()
    ci = atoms.info.get('center_atom', 0)

    # PCA on cluster centroid for stable orientation; display centred on center atom.
    c = pos - pos.mean(axis=0)
    _, _, Vt = np.linalg.svd(c, full_matrices=True)
    plane = [i for i in range(3) if i != view]
    xy = pos @ Vt[plane].T
    xy -= xy[ci]   # shift so center atom sits at (0, 0)

    radii = np.array([covalent_radii[atomic_numbers[s]] for s in syms])

    for k, sym in enumerate(syms):
        r = radii[k]
        ax.add_patch(Circle(xy[k], radius=r,
                            color=_ELEM_COLOR.get(sym, '#BBBBBB'),
                            ec='#111', lw=0.65, zorder=2))
        ax.add_patch(Circle((xy[k][0] - 0.3 * r, xy[k][1] + 0.3 * r),
                            radius=0.18 * r,
                            color='white', alpha=0.6, lw=0, zorder=3))

    if view_range is None:
        half = float(np.abs(xy).max() + radii.max())
        lim = half * (1 + margin)
    else:
        lim = float(view_range)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_rdf(r, g, insets=None, ax=None,
             inset_w=0.13, inset_h=0.38, inset_y=0.75, inset_margin=0.2,
             atomic_scale=None, figsize=(10, 5), **line_kw):
    """Plot g(r) and optionally add per-peak structure insets.

    Parameters
    ----------
    r, g : array-like
        RDF arrays as returned by :func:`rdf`.
    insets : dict, optional
        ``{peak_r: spec}`` mapping.  Each value is one of:

        - an ``ase.Atoms`` object (cluster placed at the auto-computed x)
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
    from ase import Atoms as _AseAtoms

    r = np.asarray(r)
    g = np.asarray(g)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    line_kw.setdefault('lw', 1.5)
    ax.plot(r, g, **line_kw)
    ax.set_xlabel('')
    ax.set_ylabel('')

    if insets:
        from ase.data import covalent_radii, atomic_numbers as _anum
        r_min, r_max = float(r[0]), float(r[-1])

        # Parse specs and load atoms in one pass.
        parsed = []
        for rp, spec in insets.items():
            if isinstance(spec, dict):
                fspec = spec['file']
                s = spec.get('scale', 1.0)
                w = spec.get('w', inset_w) * s
                h = spec.get('h', inset_h) * s
                y = spec.get('y', inset_y)
                m = spec.get('margin', inset_margin)
                view = spec.get('view', 2)
            else:
                fspec, w, h, y, m, view = spec, inset_w, inset_h, inset_y, inset_margin, 2

            if isinstance(fspec, _AseAtoms):
                atoms = fspec
            else:
                path, idx = _parse_fspec(fspec)
                atoms = ase_read(path, index=idx)
            parsed.append((rp, atoms, w, h, y, m, view))

        # When atomic_scale is set, unify the view range across all insets (so the
        # central atom appears the same physical size in every inset) and scale the
        # inset box by atomic_scale. Keeping the view range fixed at the largest
        # natural extent avoids clipping; a larger box makes atoms appear bigger.
        if atomic_scale is not None:
            max_nat = 0.0
            for _, atoms, _, _, _, m, view in parsed:
                pos = atoms.get_positions()
                ci = atoms.info.get('center_atom', 0)
                c = pos - pos.mean(axis=0)
                _, _, Vt = np.linalg.svd(c, full_matrices=True)
                plane = [i for i in range(3) if i != view]
                xy_p = pos @ Vt[plane].T
                xy_p -= xy_p[ci]
                radii_a = np.array([covalent_radii[_anum[s]] for s in atoms.get_chemical_symbols()])
                half = float(np.abs(xy_p).max() + radii_a.max())
                max_nat = max(max_nat, half * (1 + m))
            unified_vr = max_nat
        else:
            unified_vr = None

        for rp, atoms, w, h, y, m, view in parsed:
            w_eff = w * atomic_scale if atomic_scale is not None else w
            h_eff = h * atomic_scale if atomic_scale is not None else h
            xf = (float(rp) - r_min) / (r_max - r_min)
            xf = float(np.clip(xf - w_eff / 2, 0.01, 1.0 - w_eff - 0.01))
            ins = ax.inset_axes([xf, y - h_eff / 2, w_eff, h_eff])
            ins.set_in_layout(False)
            _draw_cluster_2d(ins, atoms, margin=m, view=view, view_range=unified_vr)
            ins.set_clip_on(False)
            for patch in ins.patches:
                patch.set_clip_on(False)

    return ax
