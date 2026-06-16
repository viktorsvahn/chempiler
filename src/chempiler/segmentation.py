"""Coarse trajectory segmentation based on changes in molecular count.

Useful for identifying time windows where the chemical composition is stable
(e.g., between reaction events), which can then be used to compute per-segment
MSDs or other time-averaged properties.
"""

import numpy as np


def segment_by_molecule_count(frames, block=100, threshold=0.5):
    """Split a trajectory into segments of roughly constant molecule count.

    The trajectory is divided into blocks of *block* frames. Adjacent blocks
    whose mean molecule counts differ by more than *threshold* are treated as
    a boundary between segments.

    Parameters
    ----------
    frames : list of Frame
    block : int
        Number of frames per averaging block.
    threshold : float
        Minimum change in mean molecule count between adjacent blocks required
        to declare a new segment.

    Returns
    -------
    list of tuple of (int, int)
        Each tuple is a half-open interval ``[start, end)`` of frame indices
        belonging to one segment.

    Notes
    -----
    This is a coarse detector: it tracks the *total* molecule count, so
    isodesmic reactions (where molecule count is conserved) are not detected.
    For per-molecule reaction detection see state_engine.ligand_exchange or
    state_engine.coordination_dynamics.
    """
    signal = np.array([len(f.molecules) for f in frames])

    means = [
        signal[i * block:(i + 1) * block].mean()
        for i in range(len(signal) // block)
    ]

    segments = []
    start = 0
    for i in range(1, len(means)):
        if abs(means[i] - means[i - 1]) > threshold:
            segments.append((start, i * block))
            start = i * block

    segments.append((start, len(frames)))
    return segments


def lifetime_segments(frames, formula):
    """Return contiguous frame intervals where a molecular species is present.

    Parameters
    ----------
    frames : list of Frame
    formula : str
        Molecular formula to search for (e.g. ``"H5O3"``).

    Returns
    -------
    list of tuple of (int, int)
        Each tuple is a half-open interval ``[start, end)`` of consecutive
        frame indices during which at least one molecule with *formula* exists.

    Examples
    --------
    >>> segs = lifetime_segments(traj.frames, "H5O3")
    >>> segs
    [(102, 108), (431, 435), (891, 897)]
    >>> frames_in_first_seg = traj.frames[segs[0][0]:segs[0][1]]
    """
    segments = []
    start = None
    for i, frame in enumerate(frames):
        present = formula in frame.formula_to_mols
        if present and start is None:
            start = i
        elif not present and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(frames)))
    return segments
