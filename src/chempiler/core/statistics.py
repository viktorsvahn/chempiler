"""Block-averaging statistics for correlated time-series data.

Standard error from a single time series is underestimated when adjacent
samples are correlated. Block averaging groups samples into blocks larger than
the autocorrelation time so that block means are approximately independent,
giving an unbiased standard error estimate.
"""

import numpy as np


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
