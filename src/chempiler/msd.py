"""Mean squared displacement (MSD) for molecular species in reactive trajectories.

Molecules are tracked within their lifetime segments using minimum-image
nearest-neighbour COM matching between consecutive frames. Unwrapped
trajectories are built from cumulative per-frame MIC displacements and
combined with the standard windowed estimator.
"""

import warnings

import numpy as np
from .segmentation import lifetime_segments


def _track_segment(frames, formula, start, end):
    """Track molecule COMs across one contiguous lifetime segment.

    Returns a list of (n, 3) arrays of unwrapped COM positions in Å.
    Molecules that appear or disappear mid-segment produce shorter tracks.
    Consecutive frames are linked by minimum-image nearest-neighbour matching.
    """
    active_wrapped   = None  # (n, 3) wrapped COMs of current active molecules
    active_unwrapped = None  # (n, 3) cumulative unwrapped positions
    active_hist      = None  # list[list[ndarray]] — per-molecule position history

    completed = []

    for fi in range(start, end):
        frame   = frames[fi]
        mol_ids = frame.formula_to_mols.get(formula, [])
        if not mol_ids:
            continue

        coms = np.array([frame.coms[m] for m in mol_ids])  # (n_curr, 3)
        cell     = np.asarray(frame.atoms.get_cell())
        inv_cell = np.linalg.inv(cell)

        if active_wrapped is None:
            active_wrapped   = coms.copy()
            active_unwrapped = coms.copy()
            active_hist      = [[p.copy()] for p in coms]
            continue

        n_prev = len(active_wrapped)
        n_curr = len(coms)

        # MIC displacement from every prev COM to every curr COM: (n_curr, n_prev, 3)
        diffs      = coms[:, None, :] - active_wrapped[None, :, :]
        diffs_frac = diffs.reshape(-1, 3) @ inv_cell
        diffs_frac -= np.round(diffs_frac)
        diffs_mic  = (diffs_frac @ cell).reshape(n_curr, n_prev, 3)
        dist2      = (diffs_mic ** 2).sum(axis=2)  # (n_curr, n_prev)

        # Greedy nearest-neighbour matching (prev → curr)
        new_wrapped   = []
        new_unwrapped = []
        new_hist      = []
        matched_curr  = set()

        for i in range(n_prev):
            row = dist2[:, i].copy()
            for j_used in matched_curr:
                row[j_used] = np.inf
            j = int(np.argmin(row))
            if not np.isinf(row[j]):
                matched_curr.add(j)
                dr = diffs_mic[j, i]
                new_unwrapped.append(active_unwrapped[i] + dr)
                new_wrapped.append(coms[j])
                active_hist[i].append(active_unwrapped[i] + dr)
                new_hist.append(active_hist[i])
            else:
                if len(active_hist[i]) > 1:
                    completed.append(np.array(active_hist[i]))

        # Newly appearing molecules start fresh tracks
        for j in range(n_curr):
            if j not in matched_curr:
                new_wrapped.append(coms[j].copy())
                new_unwrapped.append(coms[j].copy())
                new_hist.append([coms[j].copy()])

        if new_wrapped:
            active_wrapped   = np.array(new_wrapped)
            active_unwrapped = np.array(new_unwrapped)
            active_hist      = new_hist
        else:
            active_wrapped = active_unwrapped = active_hist = None

    if active_hist:
        for hist in active_hist:
            if len(hist) > 1:
                completed.append(np.array(hist))

    return completed


def _windowed_msd(tracks, max_lag):
    """Standard windowed MSD estimator over a list of unwrapped COM trajectories."""
    msd_sum = np.zeros(max_lag, dtype=np.float64)
    counts  = np.zeros(max_lag, dtype=np.int64)

    for track in tracks:
        T     = len(track)
        limit = min(max_lag, T - 1)
        for lag in range(1, limit + 1):
            disp = track[lag:] - track[:T - lag]   # (T-lag, 3)
            msd_sum[lag - 1] += (disp ** 2).sum()
            counts [lag - 1] += T - lag

    with np.errstate(invalid="ignore"):
        msd_vals = np.where(counts > 0, msd_sum / counts, np.nan)

    return msd_vals, counts


def msd(frames, formula, max_lag=None, correlation_time=None, buffer=5):
    """Compute the mean squared displacement for a molecular species.

    Molecular tracks are built within each lifetime segment of *formula*
    using minimum-image nearest-neighbour COM matching between consecutive
    frames. All tracks across all segments are combined with the standard
    windowed estimator.

    Parameters
    ----------
    frames : list of Frame
    formula : str
        Molecular formula (e.g. ``"H2O"`` or ``"HO"``).
    max_lag : int, optional
        Maximum lag in frames. Defaults to half the longest lifetime segment.
    correlation_time : int, optional
        Estimated autocorrelation time of the species motion in frames. When
        given, any lifetime segment shorter than ``correlation_time * buffer``
        triggers a :class:`UserWarning` because it cannot contain enough
        independent origins to give reliable MSD statistics and may represent
        a spurious (transient, unphysical) intermediate. MSD is still computed
        over all segments — the warning is informational only.
    buffer : int
        Safety factor applied to *correlation_time* (default 5). Segments
        longer than ``correlation_time * buffer`` frames are considered
        statistically reliable.

    Returns
    -------
    lags : numpy.ndarray
        Lag times in frames (1, 2, …, max_lag).
    msd_vals : numpy.ndarray
        Mean squared displacement in Å² at each lag.
    n_samples : numpy.ndarray
        Number of displacement samples at each lag.

    Raises
    ------
    ValueError
        If *formula* is not present in any frame.
    """
    segs = lifetime_segments(frames, formula)
    if not segs:
        raise ValueError(f"{formula!r} not found in any frame.")

    if correlation_time is not None:
        threshold = correlation_time * buffer
        short_segs = [(s, e) for s, e in segs if (e - s) < threshold]
        if short_segs:
            frac = len(short_segs) / len(segs)
            warnings.warn(
                f"{len(short_segs)} of {len(segs)} '{formula}' lifetime segments "
                f"({100 * frac:.0f}%) are shorter than {threshold} frames "
                f"(correlation_time={correlation_time} × buffer={buffer}). "
                f"These may be spurious intermediates and will contribute "
                f"unreliable MSD data. Consider increasing 'persistence' in "
                f"atom_hop() or using a stricter bond cutoff.",
                UserWarning,
                stacklevel=2,
            )

    all_tracks = []
    for start, end in segs:
        all_tracks.extend(_track_segment(frames, formula, start, end))

    if not all_tracks:
        raise ValueError(f"No molecule tracks could be built for {formula!r}.")

    longest = max(end - start for start, end in segs)
    if max_lag is None:
        max_lag = longest // 2

    lags               = np.arange(1, max_lag + 1)
    msd_vals, n_samples = _windowed_msd(all_tracks, max_lag)

    return lags, msd_vals, n_samples
