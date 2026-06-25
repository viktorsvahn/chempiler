"""Van Hove self-correlation function G_s(r, τ) for molecular species."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .msd import _track_segment
from .segmentation import lifetime_segments


def van_hove(frames, formula, lags=(1, 10, 50, 200), rmax=None, dr=0.1,
             return_peaks=False, return_n=False,
             kde=False, kde_bandwidth=None,
             stitch=False, stitch_max_gap=50, stitch_max_jump=4.0):
    """Self part of the Van Hove correlation function.

    G_s(r, τ) is the probability density of finding a molecule at distance r
    from its position τ frames earlier. At short lags the distribution is
    narrow and peaked near r = 0; at long lags it broadens as molecules diffuse.

    Parameters
    ----------
    frames : list of Frame
    formula : str
        Molecular formula to track (e.g. ``"H2O"``).
    lags : iterable of int
        Lag times in frames at which to evaluate G_s.
    rmax : float, optional
        Maximum displacement distance in Å.  Defaults to the 99th percentile
        of all displacements at the largest lag, ensuring the distribution is
        not cut off for any of the requested lags.
    dr : float
        Bin width in Å.
    kde : bool
        If True, smooth each p(r) with a Gaussian kernel after histogramming.
        This is equivalent to kernel density estimation on a regular grid and
        suppresses spurious shoulders from Poisson bin noise.  Default False
        (raw histogram).
    kde_bandwidth : float or None
        Gaussian kernel standard deviation in Å.  Used only when *kde=True*.
        If None (default), Scott's rule is applied per lag:
        h = 1.06 × σ × N⁻¹/⁵, where σ is the standard deviation of the
        displacements and N is the number of samples.  A fixed value (e.g.
        ``kde_bandwidth=0.15``) applies the same width at all lags, which is
        useful for heatmaps where visual consistency across lags matters.
    stitch : bool
        If True, chain lifetime segments across proton-transfer hops using
        :func:`~chempiler.segmentation.stitch_identity_tracks` before
        computing displacements.  This makes the Grotthuss jump distance
        (~2.8 Å) visible as a shoulder at short lags in the HO van Hove.
        Default False (standard per-segment tracking).
    stitch_max_gap : int
        Passed to ``stitch_identity_tracks`` as *max_gap*.  Default 2.
    stitch_max_jump : float
        Passed to ``stitch_identity_tracks`` as *max_jump* (Å).  Default 4.0.

    Returns
    -------
    r : numpy.ndarray
        Bin centres in Å.
    p : numpy.ndarray, shape (len(lags), len(r))
        Radial displacement probability p(r, τ) = 4πr² G_s(r, τ), normalised
        so ∫ p(r) dr = 1.  Goes to zero naturally at r = 0 and peaks at
        r = √(2Dτ) for Fickian diffusion.  Free of the 1/r² artefact that
        appears when plotting G_s directly.
    return_peaks : bool
        If True, also return r_peaks (peak position of p(r) at each lag).
        For Fickian diffusion r_peak² = 2Dτ, so D = r_peaks² / (2τ).
    return_n : bool
        If True, also return n_origins (number of displacement samples per
        lag). Useful for filtering lags with poor statistics.

    Returns
    -------
    r : numpy.ndarray
        Bin centres in Å.
    p : numpy.ndarray, shape (len(lags), len(r))
        Radial displacement probability (see above).
    r_peaks : numpy.ndarray, shape (len(lags),)  — only if return_peaks=True
    n_origins : numpy.ndarray, shape (len(lags),) — only if return_n=True
    """
    if stitch:
        from .segmentation import stitch_identity_tracks
        all_tracks, _ = stitch_identity_tracks(
            frames, formula,
            max_gap=stitch_max_gap,
            max_jump=stitch_max_jump,
        )
        if not all_tracks:
            raise ValueError(f"{formula!r} not found in any frame.")
    else:
        segs = lifetime_segments(frames, formula)
        if not segs:
            raise ValueError(f"{formula!r} not found in any frame.")
        all_tracks = []
        for s, e in segs:
            all_tracks.extend(_track_segment(frames, formula, s, e))

    lags = list(lags)
    max_lag = max(lags)

    # Collect all displacements at max lag to set rmax automatically
    if rmax is None:
        all_d = []
        for track in all_tracks:
            if len(track) <= max_lag:
                continue
            d = np.linalg.norm(track[max_lag:] - track[: len(track) - max_lag], axis=1)
            all_d.extend(d.tolist())
        rmax = float(np.percentile(all_d, 99)) * 1.1 if all_d else 10.0

    edges     = np.arange(0.0, rmax + dr, dr)
    r         = 0.5 * (edges[:-1] + edges[1:])
    p         = np.zeros((len(lags), len(r)))
    n_origins = np.zeros(len(lags), dtype=int)

    for li, lag in enumerate(lags):
        disps = []
        for track in all_tracks:
            if len(track) <= lag:
                continue
            d = np.linalg.norm(track[lag:] - track[: len(track) - lag], axis=1)
            disps.extend(d.tolist())

        if disps:
            disps_arr = np.asarray(disps)
            counts, _ = np.histogram(disps_arr, bins=edges)
            n_binned  = counts.sum()
            n_origins[li] = n_binned
            if n_binned > 0:
                # p(r) = counts / (n × dr): radial probability, no 1/r² factor
                p_raw = counts / (n_binned * dr)
                if kde:
                    if kde_bandwidth is None:
                        # Scott's rule: h = 1.06 σ N^{-1/5}
                        h = 1.06 * disps_arr.std() * len(disps_arr) ** (-0.2)
                    else:
                        h = float(kde_bandwidth)
                    sigma_bins = max(h / dr, 0.5)
                    smoothed   = gaussian_filter1d(p_raw, sigma=sigma_bins)
                    total      = smoothed.sum() * dr
                    p[li]      = smoothed / total if total > 0 else smoothed
                else:
                    p[li] = p_raw

    out = (r, p)
    if return_peaks:
        def _peak(curve):
            s = gaussian_filter1d(curve, sigma=2)
            k = int(np.argmax(s))
            if 0 < k < len(s) - 1:
                # parabolic interpolation for sub-bin precision
                y0, y1, y2 = s[k - 1], s[k], s[k + 1]
                denom = y0 - 2 * y1 + y2
                delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
                return r[k] + delta * dr
            return r[k]

        r_peaks = np.array([_peak(p[i]) if n_origins[i] > 0 else np.nan
                            for i in range(len(lags))])
        out += (r_peaks,)
    if return_n:
        out += (n_origins,)
    return out


def van_hove_heatmap(r, p, lags, ax=None, cmap="viridis", normalize_rows=True,
                     show_d2=True, sg_window=9, sg_poly=3, d2_kw=None):
    """Heatmap of p(r, τ) with zero-contours of d²p/dr² overlaid.

    The heatmap shows how the radial displacement distribution evolves with
    lag time.  Zero-contours of the second derivative trace peak positions
    and shoulders (e.g. Grotthuss jumps at r ≈ 2.8 Å for OH⁻) across all
    lags simultaneously.

    Parameters
    ----------
    r : numpy.ndarray
        Bin centres in Å — first return value of :func:`van_hove`.
    p : numpy.ndarray, shape (n_lags, n_r)
        Radial displacement probability — second return value of :func:`van_hove`.
    lags : array-like, shape (n_lags,)
        Lag times passed to :func:`van_hove` (frames or physical time).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  Created if not provided.
    cmap : str
        Colormap for p(r, τ).
    normalize_rows : bool
        If True (default), normalise each lag's p(r) to its own maximum before
        mapping colour, so that narrow short-lag peaks and broad long-lag peaks
        share the same colour scale.
    show_d2 : bool
        If True (default), overlay zero-contours of d²p/dr² in white.
    sg_window : int
        Savitzky-Golay window length for the second-derivative smoothing
        (must be odd, default 9).
    sg_poly : int
        Savitzky-Golay polynomial order (default 3).
    d2_kw : dict, optional
        Extra keyword arguments forwarded to ``ax.contour`` for the d²p
        contour lines.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter

    lags  = np.asarray(lags, dtype=float)
    p     = np.asarray(p,    dtype=float)
    dr    = float(r[1] - r[0])

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    p_plot = p.copy()
    if normalize_rows:
        row_max = p_plot.max(axis=1, keepdims=True)
        row_max = np.where(row_max > 0, row_max, 1.0)
        p_plot /= row_max

    ax.pcolormesh(r, lags, p_plot, cmap=cmap, shading="auto")
    ax.set_yscale("log")

    if show_d2:
        # Savitzky-Golay second derivative; mask low-density tails to suppress noise
        d2p = np.zeros_like(p)
        for i in range(len(lags)):
            if p[i].max() > 0 and len(p[i]) >= sg_window:
                d2p[i] = savgol_filter(p[i], window_length=sg_window,
                                       polyorder=sg_poly, deriv=2, delta=dr)

        # only show contour where p is above 5% of row maximum
        row_max_full = p.max(axis=1, keepdims=True).clip(min=1e-30)
        mask = p / row_max_full < 0.05
        d2p_masked = np.where(mask, np.nan, d2p)

        kw = dict(colors="white", linewidths=0.8, alpha=0.7)
        if d2_kw:
            kw.update(d2_kw)
        try:
            ax.contour(r, lags, d2p_masked, levels=[0.0], **kw)
        except Exception:
            pass   # contour fails if all-NaN after masking

    ax.set_xlabel("r (Å)")
    ax.set_ylabel("lag")
    return ax


# ---------------------------------------------------------------------------
# Axes extension
# ---------------------------------------------------------------------------
import matplotlib.axes as _mpl_axes


def _ax_van_hove_heatmap(self, r, p, lags, **kwargs):
    return van_hove_heatmap(r, p, lags, ax=self, **kwargs)


_mpl_axes.Axes.van_hove_heatmap = _ax_van_hove_heatmap
