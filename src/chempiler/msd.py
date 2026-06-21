"""Mean squared displacement (MSD) for molecular species in reactive trajectories.

Molecules are tracked within their lifetime segments using minimum-image
nearest-neighbour COM matching between consecutive frames. Unwrapped
trajectories are built from cumulative per-frame MIC displacements and
combined with the standard windowed estimator.
"""

import warnings

import numpy as np
from .segmentation import lifetime_segments


def diffusive_window(lags, msd, window_frac=0.25, stride_frac=0.05):
    """Find the diffusive (Fickian) regime by sliding a window over log-log MSD.

    For each window position a line is fitted to ``log(MSD)`` vs ``log(t)``.
    The window whose slope α is closest to 1 *and* whose fit is most linear
    (high R²) is selected as the diffusive regime.  Windows are scored by
    ``|α − 1| / R²``; the minimum is chosen.

    Parameters
    ----------
    lags : array-like
        Lag times (any units, must be positive).
    msd : array-like
        MSD values in Å².
    window_frac : float
        Window width as a fraction of the total number of valid points
        (default 0.25 → 25 %).
    stride_frac : float
        Stride as a fraction of the window width (default 0.05 → 5 %).

    Returns
    -------
    mask : numpy.ndarray of bool
        True for every lag that falls inside the best window.
    onset : float
        Lag time at the start of the best window.
    alpha : float
        Log-log slope (≈ 1 for Fickian diffusion) in the best window.
    r2 : float
        R² of the log-log fit in the best window.
    """
    lags = np.asarray(lags, dtype=float)
    msd  = np.asarray(msd,  dtype=float)

    valid = (lags > 0) & (msd > 0) & ~np.isnan(msd)
    idx   = np.where(valid)[0]

    if len(idx) < 4:
        raise ValueError("Not enough valid MSD points to scan for diffusive regime.")

    log_t   = np.log(lags[idx])
    log_msd = np.log(msd[idx])
    N       = len(idx)

    win    = max(4, int(window_frac * N))
    stride = max(1, int(stride_frac * win))

    best_score = np.inf
    best_i     = 0

    for i in range(0, N - win + 1, stride):
        lt = log_t  [i:i + win]
        lm = log_msd[i:i + win]

        # linear fit in log-log space
        coeffs   = np.polyfit(lt, lm, 1)
        alpha    = coeffs[0]
        residuals = lm - np.polyval(coeffs, lt)
        ss_res   = (residuals ** 2).sum()
        ss_tot   = ((lm - lm.mean()) ** 2).sum()
        r2       = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        score = abs(alpha - 1.0) / max(r2, 1e-6)
        if score < best_score:
            best_score = score
            best_i     = i
            best_alpha = alpha
            best_r2    = r2

    mask = np.zeros(len(lags), dtype=bool)
    mask[idx[best_i:best_i + win]] = True
    return mask, float(lags[idx[best_i]]), float(best_alpha), float(best_r2)


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


def _track_msd(track, max_lag):
    """MSD curve for a single unwrapped COM track."""
    T      = len(track)
    limit  = min(max_lag, T - 1)
    s      = np.zeros(max_lag, dtype=np.float64)
    c      = np.zeros(max_lag, dtype=np.int64)
    for lag in range(1, limit + 1):
        disp       = track[lag:] - track[:T - lag]   # (T-lag, 3)
        s[lag - 1] = (disp ** 2).sum()
        c[lag - 1] = T - lag
    with np.errstate(invalid="ignore"):
        return np.where(c > 0, s / c, np.nan), c


def _windowed_msd(tracks, max_lag, block_size=None):
    """Standard windowed MSD estimator over a list of unwrapped COM trajectories.

    Parameters
    ----------
    block_size : int or None
        If given, split each track into non-overlapping blocks of this length
        and compute an independent MSD curve per block. Returns stderr across
        blocks as a third element. Uses negligible extra memory compared to the
        raw-sample approach.
    """
    if block_size is None:
        msd_sum = np.zeros(max_lag, dtype=np.float64)
        counts  = np.zeros(max_lag, dtype=np.int64)
        for track in tracks:
            s, c    = _track_msd(track, max_lag)
            valid   = c > 0
            msd_sum[valid] += s[valid] * c[valid]
            counts [valid] += c[valid]
        with np.errstate(invalid="ignore"):
            return np.where(counts > 0, msd_sum / counts, np.nan), counts

    # Block path: one MSD curve per block, then mean ± stderr across blocks
    block_curves = []
    for track in tracks:
        T = len(track)
        for start in range(0, T - block_size + 1, block_size):
            block = track[start:start + block_size]
            curve, _ = _track_msd(block, max_lag)
            block_curves.append(curve)

    if not block_curves:
        msd_vals, counts = _windowed_msd(tracks, max_lag, block_size=None)
        return msd_vals, counts, np.full(max_lag, np.nan)

    arr     = np.array(block_curves)          # (n_blocks, max_lag)
    n_valid = np.sum(~np.isnan(arr), axis=0)
    mean    = np.nanmean(arr, axis=0)
    with np.errstate(invalid="ignore"):
        stderr = np.where(
            n_valid > 1,
            np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n_valid),
            np.nan,
        )
    return mean, n_valid, stderr


def msd(frames, formula, max_lag=None, correlation_time=None, buffer=5,
        block_size=None, n_blocks=None, dt=None):
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
    block_size : int, optional
        Block size in frames for block-averaged uncertainty estimation.
        Mutually exclusive with *n_blocks*.
    n_blocks : int, optional
        Number of equal-length blocks to split the trajectory into.
        The block size is ``total_frames // n_blocks``.
        Mutually exclusive with *block_size*.
    dt : float, optional
        Time per frame. If given, lags are returned in the same units as *dt*
        (e.g. ``dt=5e-3`` for ps when each frame is 5 fs). Defaults to
        integer frame indices.

    Returns
    -------
    lags : numpy.ndarray
        Lag times in frames, or physical time if *dt* is given.
    msd_vals : numpy.ndarray
        Mean squared displacement in Å² at each lag.
    n_samples : numpy.ndarray
        Number of displacement samples at each lag.
    stderr : numpy.ndarray
        Standard error of the MSD at each lag via block averaging.
        Only returned when *block_size* or *n_blocks* is given.

    Raises
    ------
    ValueError
        If *formula* is not present in any frame, or if both *block_size*
        and *n_blocks* are given.
    """
    if block_size is not None and n_blocks is not None:
        raise ValueError("Specify only one of block_size or n_blocks, not both.")

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

    lags = np.arange(1, max_lag + 1, dtype=float) * (dt if dt is not None else 1)

    _block_size = None
    if block_size is not None:
        _block_size = int(block_size)
    elif n_blocks is not None:
        total_frames = sum(end - start for start, end in segs)
        _block_size  = max(1, total_frames // n_blocks)

    if _block_size is not None:
        msd_vals, n_samples, stderr = _windowed_msd(all_tracks, max_lag, block_size=_block_size)
        return lags, msd_vals, n_samples, stderr

    msd_vals, n_samples = _windowed_msd(all_tracks, max_lag)
    return lags, msd_vals, n_samples
