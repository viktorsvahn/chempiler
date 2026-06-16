"""Reaction rate and lifetime statistics from species lifetime segments."""
import numpy as np
from .segmentation import lifetime_segments


def reaction_kinetics(frames, formula, dt=1.0):
    """Return kinetic statistics for a reactive species.

    Parameters
    ----------
    frames : list of Frame
    formula : str
        Molecular formula to track (e.g. ``"HO"``).
    dt : float
        Duration of one frame in any consistent unit (e.g. seconds).
        Default 1.0 returns results in frames.

    Returns
    -------
    dict
        n_events        : int — total number of appearance events
        lifetimes       : ndarray — per-event lifetime in dt units
        mean_lifetime   : float
        median_lifetime : float
        rate            : float — events per dt unit of total simulation time
    """
    segs = lifetime_segments(frames, formula)
    lifetimes = np.array([e - s for s, e in segs], dtype=float) * dt
    total_time = len(frames) * dt

    return {
        "n_events":        len(segs),
        "lifetimes":       lifetimes,
        "mean_lifetime":   float(lifetimes.mean())     if len(lifetimes) else np.nan,
        "median_lifetime": float(np.median(lifetimes)) if len(lifetimes) else np.nan,
        "rate":            len(segs) / total_time      if total_time > 0 else np.nan,
    }
