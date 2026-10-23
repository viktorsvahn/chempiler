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
import warnings
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import numpy as np
from ase.data import chemical_symbols
from ase.geometry import find_mic
from ase.io import read as ase_read

from .selectors import resolve
from .core.statistics import blocked_stderr

# Blocking-analysis settings for every error bar in this module (see
# chempiler.core.statistics.blocked_stderr). Fixed, so results reproduce.
MIN_BLOCKS = 20
PLATEAU_SIGMA = 1.0


def _min_half_width(frames):
    """Half the smallest perpendicular cell width over all periodic frames.

    Beyond this distance the minimum-image convention no longer finds every
    pair, so g(r) is undercounted. The perpendicular width (distance between
    opposite faces, V / |a_j x a_k|) equals the box length for orthorhombic
    cells and is smaller for skewed ones. The minimum over frames covers a
    fluctuating NPT box.

    Returns
    -------
    float or None
        None when no frame is periodic with a finite cell.
    """
    half = None
    for f in frames:
        if not f.atoms.pbc.any():
            continue
        cell = np.asarray(f.atoms.get_cell())
        volume = abs(np.linalg.det(cell))
        if volume == 0.0:
            continue
        areas = np.linalg.norm(np.cross(cell[[1, 2, 0]], cell[[2, 0, 1]]), axis=1)
        h = 0.5 * volume / areas.max()
        half = h if half is None else min(half, h)
    return half


def _default_rmax(frames):
    """Largest rmax that stays within the minimum-image limit.

    Parameters
    ----------
    frames : list of Frame

    Returns
    -------
    float
        rmax in Ångström.
    """
    rmax = _min_half_width(frames)
    if rmax is None:
        raise ValueError("rmax must be given for frames without a periodic cell.")
    print(f"[RDF] rmax = {rmax:.3f} Å (auto)")
    return rmax


def _is_molecule(selector):
    """True for a formula string (e.g. ``"HO"``), which rdf() reduces to
    one centre-of-mass point per molecule; element symbols and dicts
    select atoms."""
    return isinstance(selector, str) and selector not in chemical_symbols


def _molecule_coms(frame, formula, pos, cell):
    """Centre of mass of each molecule with the given formula.

    Each molecule is unwrapped with the minimum-image convention around
    its first atom before averaging, so molecules split across the
    periodic boundary get the right centre of mass.

    Returns
    -------
    mol_ids : numpy.ndarray of int32
        Molecule index of each point.
    coms : numpy.ndarray, shape (n_mol, 3)
    """
    idx = resolve(frame, {formula: None})
    mol = frame.atom_to_mol[idx]
    mol_ids, first, inv = np.unique(mol, return_index=True, return_inverse=True)
    anchors = pos[idx[first]]
    d, _ = find_mic(pos[idx] - anchors[inv], cell, pbc=True)
    m = frame.atoms.get_masses()[idx]
    msum = np.bincount(inv, weights=m)
    shift = np.stack([np.bincount(inv, weights=m * d[:, k]) for k in range(3)], axis=1)
    return mol_ids.astype(np.int32), anchors + shift / msum[:, None]


def _points(frame, selector, pos, cell):
    """Positions for *selector*, plus the atom and molecule id of each.

    ``None`` selects every atom. Molecule points have no single atom, so
    their atom id is -1.
    """
    if selector is None:
        idx = np.arange(len(pos))
        return idx, frame.atom_to_mol[idx], pos
    if _is_molecule(selector):
        mol_ids, coms = _molecule_coms(frame, selector, pos, cell)
        return np.full(len(mol_ids), -1, dtype=np.int32), mol_ids, coms
    idx = resolve(frame, selector)
    return idx, frame.atom_to_mol[idx], pos[idx]


def _rdf_worker(frames_chunk, center, target, edges, per_frame=False):
    """Accumulate a partial RDF histogram over a chunk of frames.

    Module-level so it is picklable for ProcessPoolExecutor.

    Scalars are returned per frame, not summed, so the caller can sum them
    in one fixed order: the result is then bitwise independent of how the
    frames were split across workers. The histogram holds integer counts,
    whose float sum is exact in any order.

    Parameters
    ----------
    per_frame : bool
        Also return each frame's histogram (needed for error bars).
        Frames without any center or target get zero rows.

    Returns
    -------
    hist : numpy.ndarray
        Pair counts summed over the chunk.
    frame_norm, frame_rho, frame_ncenter : numpy.ndarray of shape (n_frames,)
        ``n_center * rho_target``, the target density, and the number of
        centres in each frame; zero where the frame has no center or target.
    frame_valid : numpy.ndarray of bool
        Frames with at least one center and one target.
    frame_hist : numpy.ndarray of shape (n_frames, n_bins) or None
        Per-frame pair counts, only when *per_frame* is True.
    """
    hist = np.zeros(len(edges) - 1, dtype=np.float64)
    frame_norm = np.zeros(len(frames_chunk))
    frame_rho = np.zeros(len(frames_chunk))
    frame_ncenter = np.zeros(len(frames_chunk))
    frame_valid = np.zeros(len(frames_chunk), dtype=bool)
    frame_hist = np.zeros((len(frames_chunk), len(edges) - 1)) if per_frame else None

    # Atom-atom pairs only skip an atom paired with itself; once either
    # side is a whole molecule, skip every pair within the same molecule.
    by_molecule = _is_molecule(center) or _is_molecule(target)

    for i, frame in enumerate(frames_chunk):
        pos = frame.atoms.get_positions()
        cell = frame.atoms.get_cell()

        c_atom, c_mol, c_pos = _points(frame, center, pos, cell)
        t_atom, t_mol, t_pos = _points(frame, target, pos, cell)
        c_idx, t_idx = (c_mol, t_mol) if by_molecule else (c_atom, t_atom)

        if len(c_idx) == 0 or len(t_idx) == 0:
            continue

        volume = abs(np.linalg.det(cell))
        rho = len(t_idx) / volume
        frame_norm[i] = len(c_idx) * rho
        frame_rho[i] = rho
        frame_ncenter[i] = len(c_idx)
        frame_valid[i] = True

        rij = (t_pos[None, :, :] - c_pos[:, None, :]).reshape(-1, 3)
        _, dists = find_mic(rij, cell, pbc=True)

        ci_grid = np.repeat(c_idx, len(t_idx))
        tj_grid = np.tile(t_idx, len(c_idx))
        h = np.histogram(dists[ci_grid != tj_grid], bins=edges)[0]
        hist += h
        if per_frame:
            frame_hist[i] = h

    return hist, frame_norm, frame_rho, frame_ncenter, frame_valid, frame_hist


def _accumulate(frames, center, target, edges, per_frame, n_workers):
    """Run :func:`_rdf_worker` over *frames*, optionally across threads.

    Returns the summed histogram and the per-frame arrays concatenated in
    time order (frame_norm, frame_rho, frame_ncenter, frame_valid,
    frame_hist).
    """
    nframes = len(frames)
    if n_workers == 1:
        parts = [_rdf_worker(frames, center, target, edges, per_frame)]
    else:
        n_chunks = min(n_workers, nframes)
        chunk_size = math.ceil(nframes / n_chunks)
        chunks = [frames[i:i + chunk_size] for i in range(0, nframes, chunk_size)]

        # pool.map keeps chunk order, so per-frame rows stay in time order.
        with ThreadPoolExecutor(max_workers=n_chunks) as pool:
            parts = list(pool.map(
                _rdf_worker, chunks, repeat(center), repeat(target), repeat(edges),
                repeat(per_frame),
            ))

    hist = sum(p[0] for p in parts)
    per = [np.concatenate([p[k] for p in parts]) for k in range(1, 5)]
    frame_hist = np.concatenate([p[5] for p in parts]) if per_frame else None
    return (hist, *per, frame_hist)


# Recorded in cache keys, so error bars from another estimator never load.
ERROR_METHOD = f"blocking(min_blocks={MIN_BLOCKS}, n_sigma={PLATEAU_SIGMA})"


def _ratio_stderr(columns, b, block_size=None):
    """Block-averaged standard error of ratios ``sum_t a_t / sum_t b_t``.

    g(r) is such a ratio: pair counts over a normalisation that can change
    from frame to frame (reactive populations, NPT volume). Linearising it
    (delta method; Grossfield et al. 2018, LiveCoMS 1, 5067, Sec. 7.4)
    gives one zero-mean series per bin, ``z_t = (a_t - ratio * b_t) /
    mean(b)``, whose mean has the same variance as the ratio. That
    variance is estimated by block averaging (their Sec. 7.3.2) with one
    block size for every column of every array in *columns*, chosen by
    blocking analysis on all of them together
    (:func:`~chempiler.core.statistics.blocked_stderr`): the slowest
    column sets it, and all share the same number of blocks.

    Parameters
    ----------
    columns : list of numpy.ndarray of shape (n_frames, n_cols)
        Per-frame numerators, e.g. per-bin and cumulative pair counts.
    b : numpy.ndarray of shape (n_frames,)
        Per-frame denominator.
    block_size : int, optional
        Block size in frames; chosen automatically when None.

    Returns
    -------
    stderrs : list of numpy.ndarray, one per array in *columns*
        NaN when there are fewer than ``MIN_BLOCKS`` frames to block.
    info : dict
        ``block_size``, ``n_blocks``, ``plateau``, and the per-column
        ``no_plateau`` mask and implied correlation time ``tau`` split like
        *columns*, from
        :func:`~chempiler.core.statistics.blocked_stderr`.
    """
    b_mean = b.mean()
    widths = [a.shape[1] for a in columns]
    if block_size is None and len(b) < MIN_BLOCKS:
        # Too short to block at all: no error bar rather than a wrong one.
        info = dict(block_size=0, n_blocks=0, plateau=False, too_short=True,
                    no_plateau=[np.ones(w, dtype=bool) for w in widths],
                    tau=[np.full(w, np.nan) for w in widths])
        return [np.full(w, np.nan) for w in widths], info
    if b_mean == 0.0:
        n = len(b)
        bs = 1 if block_size is None else int(block_size)
        info = dict(block_size=bs, n_blocks=n // bs, plateau=True,
                    no_plateau=[np.zeros(w, dtype=bool) for w in widths],
                    tau=[np.ones(w) for w in widths])
        return [np.zeros(w) for w in widths], info
    z = np.concatenate(
        [(a - (a.sum(axis=0) / b.sum())[None, :] * b[:, None]) / b_mean for a in columns],
        axis=1,
    )
    out = blocked_stderr(z, block_size, MIN_BLOCKS, PLATEAU_SIGMA)
    splits = np.cumsum(widths)[:-1]
    out["no_plateau"] = np.split(out["no_plateau"], splits)
    out["tau"] = np.split(out["tau"], splits)
    return np.split(out["stderr"], splits), out


def _warn_no_plateau(info, labels, what):
    """Warn when blocking found no plateau for some columns."""
    if info["plateau"]:
        return
    if info.get("too_short"):
        warnings.warn(
            f"{what} error bars are NaN: block averaging needs at least "
            f"{MIN_BLOCKS} frames.",
            stacklevel=3,
        )
        return
    bad = [lab for mask, labs in zip(info["no_plateau"], labels) for lab in labs[mask]]
    warnings.warn(
        f"{what} error bars may be too small: the blocking analysis found no "
        f"plateau of the standard error at {len(bad)} point(s) ({', '.join(bad[:6])}"
        f"{', ...' if len(bad) > 6 else ''}), so correlations longer than the "
        f"largest block ({info['block_size']} frames, {info['n_blocks']} blocks) "
        f"may remain. Use a longer trajectory, or independent replicas.",
        stacklevel=3,
    )


def _result_keys(integrate, errors, return_neff=False, return_tau=False):
    """Names of the arrays rdf() returns, in order."""
    keys = ['r', 'g']
    if integrate:
        keys.append('n')
    if errors:
        keys.append('g_err')
        if integrate:
            keys.append('n_err')
        if return_neff:
            keys.append('g_neff')
            if integrate:
                keys.append('n_neff')
        if return_tau:
            keys.append('g_tau')
            if integrate:
                keys.append('n_tau')
    return keys


def rdf(frames, center, target, rmax=None, dr=0.02, integrate=False, n_workers=1,
        errors=False, block_size=None, return_neff=False, return_tau=False):
    """Compute the radial distribution function g(r).

    The result is deterministic and bitwise independent of *n_workers*.

    Parameters
    ----------
    frames : list of Frame
    center : str or dict
        What to measure from. An element symbol (``"O"``) or a dict
        (``{"HO": "O"}``, ``{"H2O": None}``) selects atoms, see
        selectors.resolve. A molecular formula as listed by
        ``summary()`` (``"HO"``, ``"H2O"``) gives one point per molecule
        at its centre of mass, so ``center="HO", target="H2O"`` makes n(r)
        the number of waters around the OH centre of mass. When either
        side is a formula, pairs within the same molecule are excluded.
    target : str, dict or None
        What to count around *center*; same forms as *center*. ``None``
        counts every atom, of any element: n(r) is then the number of atoms
        (not molecules) within r, and g(r) is normalised by the total
        atomic number density. With an atom *center* its own molecule's
        atoms are included (only the centre atom itself is skipped); with
        a molecular *center* they are excluded.
    rmax : float, optional
        Maximum distance in Ångström. Defaults to half the smallest
        perpendicular cell width over all frames (the minimum-image limit);
        a larger value warns.
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
    errors : bool
        If True, also return the standard error of g(r) (and of n(r) when
        *integrate* is True), by block averaging (Grossfield et al. 2018,
        LiveCoMS 1, 5067, Sec. 7.3.2). Each bin's per-frame contribution
        (linearised, their Sec. 7.4) is a time series; the frames are cut
        into equal contiguous blocks and the error is the spread of the
        block means. One block size serves every bin of g and n, chosen by
        blocking analysis (Flyvbjerg & Petersen, J. Chem. Phys. 91, 461,
        1989) as the first doubling level at which no bin's standard error
        still grows, with at least 20 blocks -- so the slowest bin sets it
        (Grossfield et al. Sec. 7.2). *frames* must be consecutive, equally
        spaced and equilibrated; with fewer than 20 frames the errors are
        NaN. Stores one histogram per frame, i.e. ``n_frames * n_bins``
        floats.
    block_size : int, optional
        Block size in frames instead of choosing it automatically. Only
        used when *errors* is True.
    return_neff : bool
        If True, also return the number of blocks behind each error bar,
        the number of independent samples, so ``n_eff - 1`` is the number
        of degrees of freedom of a Student-t confidence interval (see
        :func:`chempiler.confidence_halfwidth`). The same for every bin.
        Requires *errors*.
    return_tau : bool
        If True, also return each bin's integrated correlation time in
        frames implied by its blocked error, ``N * err**2 / s**2`` with
        ``s**2`` the frame-to-frame variance of the bin's (linearised)
        series (see :func:`chempiler.blocked_stderr`). It restates the
        error as a correlation time and is a lower bound where the blocking
        found no plateau. ``N / tau`` is the effective number of
        independent samples. Requires *errors*.

    Returns
    -------
    r : numpy.ndarray
        Bin centres in Ångström.
    g : numpy.ndarray
        g(r) values.
    n : numpy.ndarray
        Running coordination number n(r). Only returned when integrate is True.
    g_err : numpy.ndarray
        Standard error of g(r). Only returned when errors is True.
    n_err : numpy.ndarray
        Standard error of n(r), treating the mean target density as exact.
        Only returned when both integrate and errors are True.
    g_neff : numpy.ndarray
        Number of blocks behind each g_err. Only returned when return_neff
        is True.
    n_neff : numpy.ndarray
        Number of blocks behind each n_err. Only returned when
        return_neff and integrate are True.
    g_tau, n_tau : numpy.ndarray
        Correlation time in frames behind each g_err and n_err. Only
        returned when return_tau is True (n_tau also needs integrate).

    Warns
    -----
    UserWarning
        When rmax exceeds half the smallest cell width.
    UserWarning
        When errors is True and the blocking analysis finds no plateau for
        some bins: correlations longer than the largest block (with 20
        blocks) remain, and those error bars may be too small.
    """
    if return_neff and not errors:
        raise ValueError("return_neff=True requires errors=True.")
    if return_tau and not errors:
        raise ValueError("return_tau=True requires errors=True.")

    species = set().union(*(f.formulas for f in frames))
    for sel in (center, target):
        if _is_molecule(sel) and sel not in species:
            raise ValueError(
                f"{sel!r} not found in any frame. Species present: "
                f"{', '.join(sorted(species))}"
            )

    if rmax is None:
        rmax = _default_rmax(frames)
    else:
        half = _min_half_width(frames)
        if half is not None and rmax > half:
            warnings.warn(
                f"rmax = {rmax:.3f} Å exceeds half the smallest cell width "
                f"({half:.3f} Å over all frames). Pairs beyond that are missed by "
                f"the minimum-image convention, so g(r) and n(r) are too low there.",
                stacklevel=2,
            )

    edges = np.arange(0.0, rmax + dr, dr)
    r = 0.5 * (edges[:-1] + edges[1:])
    shell = 4.0 * np.pi * r**2 * dr

    hist, frame_norm, frame_rho, _, frame_valid, frame_hist = _accumulate(
        frames, center, target, edges, errors, n_workers)
    norm = frame_norm.sum()
    n_contrib = int(frame_valid.sum())

    g = np.zeros_like(hist) if norm == 0.0 else hist / (norm * shell)

    result = [r, g]
    if integrate:
        rho_avg = frame_rho.sum() / n_contrib if n_contrib else 0.0
        n = rho_avg * np.cumsum(g * shell)
        result.append(n)

    if errors:
        columns = [frame_hist / shell]
        if integrate:
            columns.append(np.cumsum(frame_hist, axis=1))
        errs, info = _ratio_stderr(columns, frame_norm, block_size)
        result.append(errs[0])
        if integrate:
            result.append(rho_avg * errs[1])
        if return_neff:
            result += [np.full(len(r), float(info["n_blocks"]))] * len(columns)
        if return_tau:
            result += list(info["tau"])
        labels = [np.array([f"g(r={x:.2f} Å)" for x in r]),
                  np.array([f"n(r={x:.2f} Å)" for x in r])][:len(columns)]
        _warn_no_plateau(info, labels, "RDF")

    return tuple(result)


def window_counts(frames, center, target, windows, n_workers=1, block_size=None,
                  return_series=False):
    """Mean number of targets per centre in distance windows, with error bars.

    For each window ``(r_lo, r_hi)`` this is ``n(r_hi) - n(r_lo)`` of
    :func:`rdf`, e.g. the number of molecules in a coordination shell
    bounded by two chosen distances. Unlike subtracting two values of n(r),
    the error is computed from the per-frame count *in* the window, so the
    strong correlation between n(r_lo) and n(r_hi) is handled correctly.
    Same normalisation and error estimator as :func:`rdf`; a window
    starting at 0 reproduces its n(r) and n_err.

    Parameters
    ----------
    frames : list of Frame
        Consecutive, equally spaced, equilibrated frames.
    center, target : str, dict or None
        As in :func:`rdf`.
    windows : sequence of (float, float)
        ``(r_lo, r_hi)`` pairs in Ångström, ``0 <= r_lo < r_hi``. Pairs
        with r_lo <= d < r_hi are counted. Windows may overlap.
    n_workers : int
        Number of parallel threads, as in :func:`rdf`.
    block_size : int, optional
        Block size in frames for the block-averaged errors, instead of
        choosing it automatically (one for all windows, as in :func:`rdf`).
    return_series : bool
        If True, also return the count per centre in every frame, for the
        distribution of the coordination number (its spread, P(CN = k))
        and for any other error analysis (blocking, bootstrap, split-half).

    Returns
    -------
    cn, cn_err, cn_neff : numpy.ndarray of shape (n_windows,)
        Mean count per centre, its standard error, and the number of
        blocks behind it (degrees of freedom ``cn_neff - 1``).
    series : numpy.ndarray of shape (n_frames, n_windows)
        Count in each window per centre, per frame; with a single centre
        (one OH) these are the integer instantaneous coordination numbers.
        NaN in frames without a centre. Its mean equals *cn* up to the
        density normalisation (identical at constant volume and number of
        targets). Only returned when return_series is True.

    Warns
    -----
    UserWarning
        When r_hi exceeds half the smallest cell width, or when the
        blocking analysis finds no plateau for some window.
    """
    windows = np.asarray(windows, dtype=float).reshape(-1, 2)
    if np.any(windows[:, 0] < 0) or np.any(windows[:, 0] >= windows[:, 1]):
        raise ValueError("every window must satisfy 0 <= r_lo < r_hi.")

    species = set().union(*(f.formulas for f in frames))
    for sel in (center, target):
        if _is_molecule(sel) and sel not in species:
            raise ValueError(
                f"{sel!r} not found in any frame. Species present: "
                f"{', '.join(sorted(species))}"
            )

    half = _min_half_width(frames)
    if half is not None and windows[:, 1].max() > half:
        warnings.warn(
            f"A window reaches {windows[:, 1].max():.3f} Å, beyond half the "
            f"smallest cell width ({half:.3f} Å); pairs beyond that are missed "
            f"by the minimum-image convention, so its count is too low.",
            stacklevel=2,
        )

    # Histogram on every distinct boundary; each window is then a sum of
    # consecutive bins.
    edges = np.unique(windows)
    nframes = len(frames)
    _, frame_norm, frame_rho, ncenter, frame_valid, frame_hist = _accumulate(
        frames, center, target, edges, True, n_workers)
    n_contrib = int(frame_valid.sum())
    if n_contrib == 0 or frame_norm.sum() == 0.0:
        z = np.zeros(len(windows))
        out = (z, z.copy(), np.full(len(windows), float(nframes)))
        return out + (np.full((nframes, len(windows)), np.nan),) if return_series else out
    rho_avg = frame_rho.sum() / n_contrib

    lo = np.searchsorted(edges, windows[:, 0])
    hi = np.searchsorted(edges, windows[:, 1])
    cum = np.concatenate([np.zeros((nframes, 1)), np.cumsum(frame_hist, axis=1)], axis=1)
    a = cum[:, hi] - cum[:, lo]                     # per-frame pair count in each window

    ratio = a.sum(axis=0) / frame_norm.sum()
    (err,), info = _ratio_stderr([a], frame_norm, block_size)
    _warn_no_plateau(info, [np.array([f"{lo_:.2f}-{hi_:.2f} Å" for lo_, hi_ in windows])],
                     "Window count")
    out = (rho_avg * ratio, rho_avg * err, np.full(len(windows), float(info["n_blocks"])))
    if return_series:
        with np.errstate(invalid="ignore", divide="ignore"):
            series = a / ncenter[:, None]
        series[ncenter == 0] = np.nan
        out += (series,)
    return out


def load_rdf(cache_file):
    """Load a previously cached g(r) (and n(r)) from a JSON file.

    Companion to ``ChempilerTrajectory.rdf(cache_file=...)``: lets you
    reuse an already-computed RDF -- e.g. to re-run :func:`rdf_shells`
    with different ``peak_min_g``/``smooth_sigma``, or replot -- purely
    from the tabulated r/g/n data, with no trajectory object involved and
    no need to rebuild or even load the underlying (potentially very
    expensive to reconstruct) trajectory topology.

    Parameters
    ----------
    cache_file : str or Path
        Path to a JSON file written by a prior
        ``ChempilerTrajectory.rdf(..., cache_file=...)`` call.

    Returns
    -------
    r, g : numpy.ndarray
    n : numpy.ndarray
        Only returned if the cache was computed with ``integrate=True``.
    g_err, n_err : numpy.ndarray
        Only returned if the cache was computed with ``errors=True``
        (``n_err`` only with ``integrate=True`` as well).
    g_neff, n_neff : numpy.ndarray
        Only returned if the cache was computed with ``return_neff=True``.
    g_tau, n_tau : numpy.ndarray
        Only returned if the cache was computed with ``return_tau=True``.
    """
    import json
    with open(cache_file, 'r') as f:
        data = json.load(f)
    return tuple(np.array(data[k]) for k in _result_keys(True, True, True, True) if k in data)


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
        Shell boundaries (Å), length ``len(peaks) + 1``. Starts at 0.0;
        ``troughs[i+1]`` is the minimum of g(r) after ``peaks[i]``. When g(r)
        has no minimum after the last peak before rmax, that shell is open
        and its outer boundary is the last r of the grid.
    cns : np.ndarray or None
        Cumulative coordination number at the outer boundary of each shell,
        i.e. ``cns[i] = n(troughs[i+1])``, length ``len(peaks)``. NaN for an
        open last shell, where no CN is defined. ``None`` when no
        coordination array was supplied. Per-shell (annular) CNs are
        ``np.diff(cns)`` if needed.
    g_min : np.ndarray
        g(r) at each outer boundary, length ``len(peaks)``; NaN for an open
        shell. A value near 1 means a shallow minimum and an ill-defined CN.
    cn_spread : np.ndarray or None
        Half the change in n(r) when the boundary moves by ±``cn_delta``:
        ``(n(t + δ) - n(t - δ)) / 2``. How sensitive each CN is to where
        the shell is cut. ``None`` when no coordination array was supplied.
    """
    peaks: np.ndarray
    troughs: np.ndarray
    cns: "np.ndarray | None"
    g_min: "np.ndarray | None" = None
    cn_spread: "np.ndarray | None" = None


def _refine_minimum(r, g, i, half_width):
    """Refine a grid minimum at index *i* with a parabola fitted to raw g(r).

    Fits over ``r[i] ± half_width``. Falls back to ``r[i]`` when the window
    has fewer than 3 points, the fit is not convex, or its vertex lies
    outside the window.
    """
    if half_width <= 0.0:
        return float(r[i])
    m = np.abs(r - r[i]) <= half_width
    if m.sum() < 3:
        return float(r[i])
    a, b, _ = np.polyfit(r[m] - r[i], g[m], 2)
    if a <= 0.0:
        return float(r[i])
    x = -b / (2.0 * a)
    if abs(x) > half_width:
        return float(r[i])
    return float(r[i] + x)


def rdf_shells(r, g, n=None, *, peak_min_g=0.5, smooth_sigma=0.0,
               refine_width=0.15, cn_delta=0.1):
    """Detect coordination shells in a pre-computed g(r).

    Peaks and minima are located on g(r) (optionally smoothed); each minimum
    is then refined with a parabola fitted to the raw g(r), and the CN is
    read from the unsmoothed n(r) at that point -- the usual first-minimum
    convention (as in e.g. TRAVIS).

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
        peak/trough detection.  ``0`` disables smoothing (default). The
        ends are padded with their edge values, not zeros, so the tail of
        g(r) is not pulled down into a spurious peak near rmax.
    refine_width : float
        Half-width in Å of the window for the parabolic refinement of each
        minimum on the raw g(r). ``0`` keeps the grid point.
    cn_delta : float
        Boundary shift in Å used for ``cn_spread``.

    Returns
    -------
    ShellInfo
        ``.peaks`` — peak positions; ``.troughs`` — shell boundaries
        (length ``len(peaks) + 1``, starts at 0.0); ``.cns[i]`` —
        cumulative CN at ``troughs[i+1]`` (NaN for an open last shell);
        ``.g_min`` and ``.cn_spread`` — how well defined each boundary is.
        ``cns`` and ``cn_spread`` are ``None`` when *n* is not supplied.
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
        g_det = np.convolve(np.pad(g, half_w, mode='edge'), kernel, mode='valid')
    else:
        g_det = g

    valid = np.flatnonzero(r > 0.5)
    v = valid[1:-1]
    is_peak = (
        (g_det[v] > g_det[v - 1]) &
        (g_det[v] > g_det[v + 1]) &
        (g_det[v] > peak_min_g)
    )
    peak_idx = v[is_peak]
    peaks = r[peak_idx].copy()

    # Grid index of the minimum after each peak; None for an open last shell.
    min_idx = []
    for k, i in enumerate(peak_idx):
        if k + 1 < len(peak_idx):
            j = int(peak_idx[k + 1])
            min_idx.append(i + int(np.argmin(g_det[i:j + 1])))
        else:
            m = i + int(np.argmin(g_det[i:]))
            # Only a real minimum if g(r) rises again after it before rmax.
            min_idx.append(m if m < len(r) - 1 and g_det[m + 1:].max() > g_det[m] else None)

    troughs = [0.0]
    g_min = []
    for m in min_idx:
        if m is None:
            troughs.append(float(r[-1]))
            g_min.append(np.nan)
        else:
            t = _refine_minimum(r, g, m, refine_width)
            troughs.append(t)
            g_min.append(float(np.interp(t, r, g)))
    troughs = np.array(troughs)
    g_min = np.array(g_min)

    if n is not None:
        n_arr = np.asarray(n, dtype=float)
        cns = np.interp(troughs[1:], r, n_arr)
        cn_spread = 0.5 * (np.interp(troughs[1:] + cn_delta, r, n_arr)
                           - np.interp(troughs[1:] - cn_delta, r, n_arr))
        open_shell = np.isnan(g_min)
        cns[open_shell] = np.nan
        cn_spread[open_shell] = np.nan
    else:
        cns = cn_spread = None

    return ShellInfo(peaks=peaks, troughs=troughs, cns=cns,
                     g_min=g_min, cn_spread=cn_spread)


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

from .colours import elem_color as _elem_color, elem_radius as _elem_radius


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
    from .colours import draw_atom_2d
    pos = atoms.get_positions()
    syms = atoms.get_chemical_symbols()
    ci = atoms.info.get('center_atom', 0)

    # PCA on cluster centroid for stable orientation; display centred on center atom.
    c = pos - pos.mean(axis=0)
    _, _, Vt = np.linalg.svd(c, full_matrices=True)
    plane = [i for i in range(3) if i != view]
    xy = pos @ Vt[plane].T
    xy -= xy[ci]   # shift so center atom sits at (0, 0)

    radii = np.array([_elem_radius(s) for s in syms])

    for k, sym in enumerate(syms):
        draw_atom_2d(ax, float(xy[k, 0]), float(xy[k, 1]), sym)

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
             inset_w=0.325, inset_h=0.95, inset_y=0.75, inset_margin=0.2,
             atomic_scale=1.0, figsize=(10, 5), g_err=None, err_alpha=0.3, **line_kw):
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
    g_err : array-like, optional
        Standard error of g(r), as returned by ``rdf(..., errors=True)``.
        Drawn as a shaded band of ±g_err around the line.
    err_alpha : float
        Opacity of the error band.
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
    (line,) = ax.plot(r, g, **line_kw)
    if g_err is not None:
        g_err = np.asarray(g_err)
        ax.fill_between(r, g - g_err, g + g_err, color=line.get_color(),
                        alpha=err_alpha, lw=0)
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


# ---------------------------------------------------------------------------
# Axes extension
# ---------------------------------------------------------------------------
import matplotlib.axes as _mpl_axes


def _ax_plot_rdf(self, r, g, **kwargs):
    return plot_rdf(r, g, ax=self, **kwargs)


_mpl_axes.Axes.plot_rdf = _ax_plot_rdf
