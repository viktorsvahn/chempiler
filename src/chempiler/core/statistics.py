"""Block-averaging statistics for correlated time-series data.

Standard error from a single time series is underestimated when adjacent
samples are correlated. Block averaging groups samples into blocks larger than
the autocorrelation time so that block means are approximately independent,
giving an unbiased standard error estimate.
"""

import warnings

import numpy as np


def _acf_fft(x):
    """Normalised autocorrelation function of a time series via FFT.

    Parameters
    ----------
    x : numpy.ndarray
        1-D time series, or a 2-D array whose columns are independent
        time series (time along axis 0).

    Returns
    -------
    numpy.ndarray of the same shape as x
        rho(t) for t = 0 .. len(x)-1, with rho(0) = 1 (per column).
    """
    n = len(x)
    # Zero-pad to the next power of two (>= 2n) so the circular
    # convolution done by the FFT does not wrap the series onto itself.
    n_pad = 1
    while n_pad < 2 * n:
        n_pad *= 2

    f = np.fft.fft(x - x.mean(axis=0), n=n_pad, axis=0)
    acf = np.fft.ifft(f * np.conjugate(f), axis=0)[:n].real
    acf /= acf[0]
    return acf


def _auto_window(taus, c):
    """Sokal's adaptive window: smallest M with M >= c * tau(M)."""
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return int(np.argmin(m))
    return len(taus) - 1


def estimate_autocorrelation_time(x, c=5.0, tol=50.0, quiet=False):
    """Estimate the integrated autocorrelation time of a 1-D time series.

    Uses the FFT-computed autocorrelation function with Sokal's automatic
    windowing (the same method behind ``emcee.autocorr.integrated_time``;
    Sokal 1997; Goodman & Weare 2010): the running estimate
    ``tau(M) = 1 + 2 * sum_{t=1}^{M} rho(t)`` is evaluated at every window
    size M, and the smallest M with ``M >= c * tau(M)`` is chosen. This
    balances the bias of a too-small window against the noise of a
    too-large one (rho(t) is dominated by statistical noise at large lag).

    An estimate is only trustworthy when the series is much longer than
    the correlation time it implies — short of that, the tail of the ACF
    used to pick the window is itself just noise, and the "automatic"
    window becomes unstable. This function enforces Sokal's rule of thumb,
    ``len(x) >= tol * tau``, and raises by default rather than silently
    returning a number that cannot be trusted.

    Parameters
    ----------
    x : array-like
        1-D time series (one value per frame). Must be (at least
        approximately) stationary — e.g. drawn from an equilibrated
        trajectory, not one still relaxing toward equilibrium.
    c : float
        Window constant for Sokal's criterion (default 5, the standard
        choice). Larger values give a more conservative (larger, lower
        variance) window.
    tol : float
        Minimum required ratio of series length to estimated tau for the
        estimate to be accepted (default 50).
    quiet : bool
        If False (default), raise ``ValueError`` when the reliability
        criterion is not met. If True, emit a ``UserWarning`` instead and
        return the (unreliable) estimate anyway.

    Returns
    -------
    float
        Estimated autocorrelation time, in frames (>= 1).

    Raises
    ------
    ValueError
        If ``x`` is not 1-D, or if the reliability criterion
        ``len(x) >= tol * tau`` is not met and ``quiet`` is False.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x must be a 1-D time series, got shape {x.shape}.")
    n = len(x)
    if n < 2:
        raise ValueError("x must have at least 2 samples.")

    acf = _acf_fft(x)
    taus = 2.0 * np.cumsum(acf) - 1.0
    window = _auto_window(taus, c)
    tau = max(float(taus[window]), 1.0)

    if n < tol * tau:
        msg = (
            f"Unreliable autocorrelation-time estimate: n={n} frames, "
            f"tau≈{tau:.1f} frames, but n >= tol*tau = {tol * tau:.0f} is "
            f"required (tol={tol:g}). Run a longer trajectory, or pass "
            f"quiet=True to accept this estimate anyway."
        )
        if not quiet:
            raise ValueError(msg)
        warnings.warn(msg)

    return tau


def integrated_autocorrelation_times(x, c=5.0, chunk=32):
    """Integrated autocorrelation time of every column of a 2-D series.

    Column-wise counterpart of :func:`estimate_autocorrelation_time`, for
    quantities with many components per frame (e.g. the bins of a
    histogram). Same Sokal-windowed FFT estimator, but without the
    reliability check -- the caller decides how to report that.

    Parameters
    ----------
    x : array-like of shape (n_frames, n_columns)
        One time series per column, time along axis 0.
    c : float
        Window constant for Sokal's criterion (default 5).
    chunk : int
        Number of columns transformed per FFT call. Bounds memory to about
        ``4 * n_frames * chunk`` complex values.

    Returns
    -------
    numpy.ndarray of shape (n_columns,)
        Autocorrelation time of each column, in frames (>= 1). Constant
        columns (zero variance) get 1.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D (n_frames, n_columns), got shape {x.shape}.")
    if len(x) < 2:
        raise ValueError("x must have at least 2 samples.")

    taus = np.ones(x.shape[1])
    live = np.flatnonzero(x.var(axis=0) > 0)
    for start in range(0, len(live), chunk):
        cols = live[start:start + chunk]
        acf = _acf_fft(x[:, cols])
        for col, rho in zip(cols, acf.T):
            t = 2.0 * np.cumsum(rho) - 1.0
            taus[col] = max(float(t[_auto_window(t, c)]), 1.0)
    return taus


def split_blocks(n_frames, tau_corr):
    """Partition frame indices into non-overlapping blocks.

    Parameters
    ----------
    n_frames : int
        Total number of frames in the time series.
    tau_corr : float
        Autocorrelation time in frames. The block size is ``int(tau_corr)``.

    Returns
    -------
    numpy.ndarray of shape (n_blocks, block_size)
        Frame indices grouped by block.

    Raises
    ------
    ValueError
        If fewer than two complete blocks fit in the trajectory.
    """
    block_size = int(tau_corr)
    n_blocks = n_frames // block_size

    if n_blocks < 2:
        raise ValueError(
            f"Too few blocks for statistics: need ≥ 2 blocks, got {n_blocks} "
            f"(n_frames={n_frames}, tau_corr={tau_corr})"
        )

    return np.arange(n_frames).reshape(n_blocks, block_size)


def block_average(values, tau_corr):
    """Compute the block-averaged mean and standard error of a time series.

    Parameters
    ----------
    values : array-like of float
        1-D time series (one value per frame).
    tau_corr : float
        Autocorrelation time in frames (sets the block size).

    Returns
    -------
    dict with keys:
        mean     : float — sample mean over all blocks.
        stderr   : float — standard error (std of block means / sqrt(n_blocks)).
        n_blocks : int   — number of blocks used.
    """
    blocks = split_blocks(len(values), tau_corr)

    means = np.asarray([np.mean(values[b]) for b in blocks])

    return {
        "mean": float(means.mean()),
        "stderr": float(means.std(ddof=1) / np.sqrt(len(means))),
        "n_blocks": len(means),
    }


def blocking_levels(x, min_blocks=20):
    """Standard error of the mean at every block size of a doubling sequence.

    The blocking analysis of Flyvbjerg & Petersen (J. Chem. Phys. 91, 461,
    1989): the series is cut into ``M = n_frames // B`` contiguous blocks
    of ``B = 1, 2, 4, ...`` frames (frames beyond ``M * B`` are dropped),
    and the standard error of the mean is estimated from the block means
    as ``s(x̄_b) / sqrt(M)`` (Grossfield et al. 2018, LiveCoMS 1, 5067,
    Sec. 7.3.2, Eqs. 13-14). It grows with B while blocks are shorter than
    the correlation time and levels off once the block means are
    independent.

    Parameters
    ----------
    x : array-like of shape (n_frames,) or (n_frames, n_columns)
        One time series per column, time along axis 0.
    min_blocks : int
        Smallest number of blocks a level may have (default 20). Fewer
        blocks give a standard error too noisy to judge a plateau by.

    Returns
    -------
    block_sizes, n_blocks : numpy.ndarray of shape (n_levels,)
    stderr : numpy.ndarray of shape (n_levels, n_columns)
        Standard error of the mean at each level (one column for 1-D *x*).
    stderr_err : numpy.ndarray of shape (n_levels, n_columns)
        Standard error of that standard error, ``stderr / sqrt(2 (M - 1))``
        (Flyvbjerg & Petersen Eq. 28).

    Raises
    ------
    ValueError
        If the series has fewer than ``min_blocks`` frames.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"x must be 1-D or 2-D, got shape {x.shape}.")
    if min_blocks < 2:
        raise ValueError(f"min_blocks must be at least 2, got {min_blocks}.")
    n = len(x)
    if n < min_blocks:
        raise ValueError(f"Need at least min_blocks={min_blocks} frames, got {n}.")

    sizes, counts, se = [], [], []
    b = 1
    while n // b >= min_blocks:
        m = n // b
        means = x[:m * b].reshape(m, b, -1).mean(axis=1)
        sizes.append(b)
        counts.append(m)
        se.append(means.std(axis=0, ddof=1) / np.sqrt(m))
        b *= 2
    sizes, counts, se = np.array(sizes), np.array(counts), np.array(se)
    return sizes, counts, se, se / np.sqrt(2.0 * (counts[:, None] - 1))


def plateau_level(stderr, stderr_err, n_sigma=1.0):
    """First blocking level after which the standard error stops growing.

    Level k is accepted when the inverse-variance weighted mean ``m`` of the
    standard errors at all later levels is not significantly larger:
    ``m - stderr[k] <= n_sigma * sqrt(err_m**2 + err[k]**2)``, with
    ``err_m = (sum 1/err[j]**2) ** -0.5``. Larger blocks then find no
    larger error -- the plateau of Flyvbjerg & Petersen 1989 (Grossfield et
    al. 2018, Sec. 7.3.2). Averaging the later levels, rather than testing
    each, keeps noise in any single level from moving the choice, while a
    standard error that keeps creeping up still fails the test. The last
    level has no later level to confirm it, so a column accepted only
    there has no plateau.

    Parameters
    ----------
    stderr, stderr_err : numpy.ndarray of shape (n_levels, n_columns)
        From :func:`blocking_levels`.
    n_sigma : float
        Tolerance in combined standard errors (default 1). Smaller is more
        conservative: a slowly creeping standard error is less easily
        mistaken for a plateau.

    Returns
    -------
    level : numpy.ndarray of int, shape (n_columns,)
        Index of the accepted level; the last level where there is no
        plateau.
    plateau : numpy.ndarray of bool, shape (n_columns,)
        Whether a confirmed plateau was found. Columns with zero variance
        count as having one, at level 0.
    """
    se = np.asarray(stderr, dtype=float)
    err = np.asarray(stderr_err, dtype=float)
    n_levels = len(se)
    # A zero error (constant block means) carries no weight; columns that
    # are constant throughout are handled below.
    with np.errstate(divide="ignore"):
        w = np.where(err > 0, 1.0 / err**2, 0.0)
    ok = np.zeros(se.shape, dtype=bool)
    ok[-1] = True                                    # unconfirmed fallback
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in range(n_levels - 1):
            w_sum = w[k + 1:].sum(axis=0)
            m = (w[k + 1:] * se[k + 1:]).sum(axis=0) / w_sum
            ok[k] = m - se[k] <= n_sigma * np.sqrt(1.0 / w_sum + err[k] ** 2)
    level = np.argmax(ok, axis=0)                    # first accepted level
    plateau = level < n_levels - 1
    flat = (se == 0).all(axis=0)
    level[flat] = 0
    plateau[flat] = True
    return level, plateau


def blocked_stderr(x, block_size=None, min_blocks=20, n_sigma=1.0):
    """Standard error of the mean of correlated series by block averaging.

    Block averaging as in Grossfield et al. 2018 (LiveCoMS 1, 5067,
    Sec. 7.3.2): the series is cut into ``M`` contiguous blocks of equal
    size B and the standard error is ``s(x̄_b) / sqrt(M)`` (their Eq. 14),
    with ``M - 1`` degrees of freedom for a Student-t interval (Sec. 7.5;
    see :func:`confidence_halfwidth`). Frames beyond ``M * B`` are dropped.

    When *block_size* is None it is chosen by blocking analysis
    (Flyvbjerg & Petersen 1989): the first doubling level at which the
    standard error stops growing (:func:`plateau_level`). For several
    columns, one block size serves all of them -- the largest any column
    needs -- so the slowest column sets it (Grossfield et al. Sec. 7.2:
    track the slowest relevant degree of freedom), and all columns share
    the same degrees of freedom. The procedure is deterministic.

    Parameters
    ----------
    x : array-like of shape (n_frames,) or (n_frames, n_columns)
        Consecutive, equally spaced samples, time along axis 0.
    block_size : int, optional
        Block size in frames. Chosen automatically when None.
    min_blocks : int
        Smallest number of blocks allowed (default 20).
    n_sigma : float
        Plateau tolerance, see :func:`plateau_level` (default 1).

    Returns
    -------
    dict with keys
        stderr : numpy.ndarray of shape (n_columns,), or float for 1-D *x*
        block_size : int
        n_blocks : int
        plateau : bool -- every column reached a confirmed plateau (always
            True when *block_size* is given).
        no_plateau : numpy.ndarray of bool -- columns without one.
        tau : numpy.ndarray, or float for 1-D *x* -- integrated correlation
            time in frames implied by the blocked error,
            ``N * stderr**2 / s**2`` over the ``N = n_blocks * block_size``
            frames used (the statistical inefficiency; Grossfield et al.
            Eq. 11-12 solved for it). Not an independent estimate: it
            restates the blocked error in the language of correlation
            times, and is a lower bound wherever there is no plateau. 1 for
            constant columns.
        n_ind : numpy.ndarray, or float for 1-D *x* -- effective number of
            independent samples, ``N / tau``.

    Raises
    ------
    ValueError
        If *block_size* leaves fewer than ``min_blocks`` blocks.
    """
    x = np.asarray(x, dtype=float)
    one_d = x.ndim == 1
    if one_d:
        x = x[:, None]
    n = len(x)

    if block_size is None:
        sizes, _, se, err = blocking_levels(x, min_blocks)
        level, ok = plateau_level(se, err, n_sigma)
        block_size = int(sizes[level.max()])
        no_plateau = ~ok
    else:
        block_size = int(block_size)
        if block_size < 1 or n // block_size < min_blocks:
            raise ValueError(
                f"block_size={block_size} leaves {n // max(block_size, 1)} blocks "
                f"of {n} frames; at least min_blocks={min_blocks} are needed."
            )
        no_plateau = np.zeros(x.shape[1], dtype=bool)

    m = n // block_size
    used = x[:m * block_size]
    means = used.reshape(m, block_size, -1).mean(axis=1)
    stderr = means.std(axis=0, ddof=1) / np.sqrt(m)
    var = used.var(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = np.where(var > 0, len(used) * stderr**2 / var, 1.0)
    n_ind = len(used) / tau
    scalar = (lambda a: float(a[0])) if one_d else (lambda a: a)
    return {
        "stderr": scalar(stderr),
        "block_size": block_size,
        "n_blocks": m,
        "plateau": not no_plateau.any(),
        "no_plateau": no_plateau,
        "tau": scalar(tau),
        "n_ind": scalar(n_ind),
    }


def confidence_halfwidth(stderr, n_eff, level=0.95):
    """Half-width of a Student-t confidence interval from correlated data.

    *n_eff* is the number of independent samples behind *stderr*: the
    number of blocks for a block-averaged error (:func:`blocked_stderr`),
    or ``n_frames / tau`` for an autocorrelation-corrected one. The
    interval is ``mean ± t_{(1+level)/2}(n_eff - 1) * stderr`` (coverage
    factor from the Student-t distribution; Grossfield et al. 2018,
    LiveCoMS 1, 5067, Sec. 7.5).

    Parameters
    ----------
    stderr : array-like
        Standard error(s), e.g. ``g_err`` from ``rdf(..., errors=True)``.
    n_eff : array-like
        Effective sample size(s), same shape as *stderr* (or broadcastable),
        e.g. ``g_neff`` from ``rdf(..., errors=True, return_neff=True)``.
    level : float
        Confidence level, between 0 and 1 (default 0.95).

    Returns
    -------
    numpy.ndarray
        Half-width of the interval. NaN where ``n_eff <= 1`` (no degrees
        of freedom left).
    """
    from scipy.stats import t

    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be between 0 and 1, got {level}.")
    stderr = np.asarray(stderr, dtype=float)
    dof = np.asarray(n_eff, dtype=float) - 1.0
    with np.errstate(invalid='ignore'):
        q = np.where(dof > 0, t.ppf(0.5 * (1.0 + level), np.where(dof > 0, dof, 1.0)), np.nan)
    return q * stderr
