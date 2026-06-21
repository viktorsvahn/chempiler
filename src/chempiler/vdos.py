"""Vibrational density of states (VDOS) from VACF log files."""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def read_vdos(logfile, col=8, dt=0.5, skip_rows=1):
    """Read a VACF column from a log file and compute the VDOS via FFT.

    A Hann window is applied before the FFT to suppress spectral leakage.

    Parameters
    ----------
    logfile : str or Path
        Whitespace-delimited log file with a VACF column.
    col : int
        Zero-based column index of the VACF (default 8).
    dt : float
        Time step in fs (default 0.5).
    skip_rows : int
        Number of header rows to skip (default 1).

    Returns
    -------
    wavenumber : numpy.ndarray
        Wavenumbers in cm⁻¹.
    vdos : numpy.ndarray
        Power spectrum in arbitrary units (raw, not smoothed).
    """
    import pandas as pd
    from scipy.fft import rfft, rfftfreq

    data = pd.read_csv(logfile, sep=r"\s+", header=None, skiprows=skip_rows)
    vacf = data[col].to_numpy(dtype=float)

    N      = len(vacf)
    vdos   = np.abs(rfft(vacf * np.hanning(N))) ** 2
    freq   = rfftfreq(N, d=dt)          # cycles / fs
    c_fs   = 2.998e-5                   # speed of light in cm / fs
    return freq / c_fs, vdos


def smooth(arr, sigma=20):
    """Gaussian smooth a 1D array.

    Parameters
    ----------
    arr : array-like
    sigma : float
        Standard deviation of the Gaussian kernel in samples.

    Returns
    -------
    numpy.ndarray
    """
    return gaussian_filter1d(np.asarray(arr, dtype=float), sigma=sigma)


def plot_vdos(wavenumber, *spectra, labels=None, smooth_sigma=20,
              normalize=True, wn_range=(0, 4500),
              show_diff=False, axes=None, **plot_kw):
    """Plot one or more VDOS spectra with optional smoothing and difference panel.

    Parameters
    ----------
    wavenumber : array-like
        Wavenumbers in cm⁻¹ — shared by all spectra.
    *spectra : array-like
        Raw VDOS arrays (e.g. from :func:`read_vdos`).  Pass two to enable
        ``show_diff``.
    labels : list of str, optional
        Legend labels, one per spectrum.
    smooth_sigma : float or None
        Gaussian smoothing width in samples.  Raw spectrum is shown faintly
        behind the smoothed curve.  Pass ``None`` to skip smoothing.
    normalize : bool
        If True (default), normalise each spectrum to its maximum within
        *wn_range* before plotting.
    wn_range : tuple (float, float)
        Wavenumber range to display in cm⁻¹.
    show_diff : bool
        If True, add a second panel showing
        ``spectra[0]_smooth − spectra[1]_smooth`` with fill_between colouring.
        Requires exactly two spectra.
    axes : matplotlib.axes.Axes or sequence of Axes, optional
        Pre-existing axes.  If *show_diff* is True, pass a length-2 sequence.
        Created automatically if not provided.
    **plot_kw
        Extra keyword arguments forwarded to the spectrum line plots.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : matplotlib.axes.Axes or list of Axes
    """
    import matplotlib.pyplot as plt

    wn   = np.asarray(wavenumber)
    mask = (wn >= wn_range[0]) & (wn <= wn_range[1])

    if show_diff and len(spectra) != 2:
        raise ValueError("show_diff requires exactly two spectra.")

    if axes is None:
        if show_diff:
            fig, axes = plt.subplots(2, 1, figsize=(9, 7),
                                     gridspec_kw={"height_ratios": [2, 1]})
        else:
            fig, ax_main = plt.subplots(figsize=(9, 4))
            axes = ax_main
    else:
        fig = (axes[0] if hasattr(axes, "__len__") else axes).get_figure()

    ax_main = axes[0] if hasattr(axes, "__len__") else axes

    normalized = []
    for i, sp in enumerate(spectra):
        sp   = np.asarray(sp, dtype=float)
        norm = sp[mask].max() if normalize and sp[mask].max() > 0 else 1.0
        sp_n = sp / norm
        normalized.append(sp_n)

        s     = smooth(sp_n, sigma=smooth_sigma) if smooth_sigma else sp_n.copy()
        label = labels[i] if labels else None
        color = f"C{i}"

        ax_main.plot(wn[mask], sp_n[mask], color=color, alpha=0.2, lw=0.8)
        ax_main.plot(wn[mask], s[mask],    color=color, lw=1.5,
                     label=label, **plot_kw)

    ax_main.set_xlim(*wn_range)
    ax_main.set_ylim(0)
    ax_main.set_ylabel("VDOS (arb. units)")
    if not show_diff:
        ax_main.set_xlabel("Wavenumber (cm⁻¹)")
    if labels:
        ax_main.legend()

    if show_diff:
        ax_diff = axes[1]
        # diff on raw (normalized) spectra, then smooth
        raw_diff = normalized[0] - normalized[1]
        diff     = smooth(raw_diff, sigma=smooth_sigma) if smooth_sigma else raw_diff
        wn_m     = wn[mask]
        diff     = diff[mask]

        ax_diff.axhline(0, color="0.5", lw=0.8, ls="--")
        ax_diff.plot(wn_m, diff, color="0.3", lw=1.2)
        ax_diff.fill_between(wn_m, diff, 0, where=diff > 0,
                              color="C0", alpha=0.3,
                              label=f"↑ {labels[0]}" if labels else "positive")
        ax_diff.fill_between(wn_m, diff, 0, where=diff < 0,
                              color="C1", alpha=0.3,
                              label=f"↑ {labels[1]}" if labels else "negative")
        ax_diff.set_xlim(*wn_range)
        ax_diff.set_xlabel("Wavenumber (cm⁻¹)")
        ax_diff.set_ylabel("Δ VDOS")
        ax_diff.legend(fontsize=9)
        plt.tight_layout()
        return fig, list(axes)

    plt.tight_layout()
    return fig, ax_main
