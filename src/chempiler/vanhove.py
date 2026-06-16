"""Van Hove self-correlation function G_s(r, τ) for molecular species."""
import numpy as np
from .msd import _track_segment
from .segmentation import lifetime_segments


def van_hove(frames, formula, lags=(1, 10, 50, 200), rmax=6.0, dr=0.1):
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
    rmax : float
        Maximum displacement distance in Å.
    dr : float
        Bin width in Å.

    Returns
    -------
    r : numpy.ndarray
        Bin centres in Å.
    G : numpy.ndarray, shape (len(lags), len(r))
        G_s(r, τ) at each lag, normalised so ∫ G_s 4π r² dr ≈ 1.
    """
    segs = lifetime_segments(frames, formula)
    if not segs:
        raise ValueError(f"{formula!r} not found in any frame.")

    all_tracks = []
    for s, e in segs:
        all_tracks.extend(_track_segment(frames, formula, s, e))

    edges = np.arange(0.0, rmax + dr, dr)
    r = 0.5 * (edges[:-1] + edges[1:])
    shell = 4.0 * np.pi * r ** 2 * dr
    lags = list(lags)
    G = np.zeros((len(lags), len(r)))

    for li, lag in enumerate(lags):
        disps = []
        n_origins = 0
        for track in all_tracks:
            if len(track) <= lag:
                continue
            d = np.linalg.norm(track[lag:] - track[: len(track) - lag], axis=1)
            disps.extend(d.tolist())
            n_origins += len(d)

        if n_origins > 0:
            counts, _ = np.histogram(disps, bins=edges)
            G[li] = counts / (n_origins * shell + 1e-30)

    return r, G
