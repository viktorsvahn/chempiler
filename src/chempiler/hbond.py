"""Hydrogen bond detection and intermittent autocorrelation."""
import numpy as np
from ase.geometry import find_mic


def _hbonds_frame(frame, r_HA=2.4):
    """Return frozenset of (donor_O, H, acceptor_O) H-bond triplets."""
    pos = frame.atoms.get_positions()
    cell = frame.atoms.get_cell()
    syms = frame.symbols

    O_idx = np.array([i for i, s in enumerate(syms) if s == "O"])
    H_list = [i for i, s in enumerate(syms) if s == "H"]

    if not H_list or not len(O_idx):
        return frozenset()

    H_arr = np.array(H_list)
    donor = np.full(len(H_list), -1, dtype=int)
    for hi, h in enumerate(H_list):
        mid = frame.atom_to_mol[h]
        if mid < 0:
            continue
        for a in frame.molecules[mid]:
            if syms[a] == "O":
                donor[hi] = a
                break

    valid = donor >= 0
    if not valid.any():
        return frozenset()

    H_arr = H_arr[valid]
    donor = donor[valid]

    # All H...O distances in one vectorised find_mic call
    H_pos = pos[H_arr]                                          # (n_H, 3)
    O_pos = pos[O_idx]                                          # (n_O, 3)
    diffs = (O_pos[None, :, :] - H_pos[:, None, :]).reshape(-1, 3)
    _, dists = find_mic(diffs, cell, pbc=True)
    dists = dists.reshape(len(H_arr), len(O_idx))              # (n_H, n_O)

    mask = (O_idx[None, :] != donor[:, None]) & (dists < r_HA)
    hi_idx, oi_idx = np.where(mask)
    return frozenset(
        (int(donor[hi]), int(H_arr[hi]), int(O_idx[oi]))
        for hi, oi in zip(hi_idx, oi_idx)
    )


def hbond_count(frames, r_HA=2.4):
    """Return number of H-bonds per frame.

    Parameters
    ----------
    frames : list of Frame
    r_HA : float
        H···acceptor-O cutoff distance in Å.

    Returns
    -------
    numpy.ndarray of int, shape (n_frames,)
    """
    return np.array([len(_hbonds_frame(f, r_HA)) for f in frames])


def hbond_acf(frames, r_HA=2.4, max_lag=200):
    """Intermittent H-bond autocorrelation function C(τ).

    C(τ) = ⟨b(0) b(τ)⟩ / ⟨b(0)⟩  averaged over all reference frames t.
    C(0) = 1 by definition; C(τ) → 0 as bonds break and reform.

    Parameters
    ----------
    frames : list of Frame
    r_HA : float
        H···acceptor-O cutoff distance in Å.
    max_lag : int
        Maximum lag in frames.

    Returns
    -------
    lags : numpy.ndarray
    C : numpy.ndarray
    """
    bond_sets = [_hbonds_frame(f, r_HA) for f in frames]

    universe = sorted(set().union(*bond_sets))
    if not universe:
        return np.arange(1, max_lag + 1), np.zeros(max_lag)

    idx = {b: i for i, b in enumerate(universe)}
    n, m = len(frames), len(universe)

    B = np.zeros((n, m), dtype=np.float32)
    for t, bs in enumerate(bond_sets):
        for b in bs:
            B[t, idx[b]] = 1.0

    n_t = B.sum(axis=1)   # bonds per frame (n,)
    C = np.zeros(max_lag)

    for lag in range(1, max_lag + 1):
        overlap = (B[: n - lag] * B[lag:]).sum(axis=1)
        denom = n_t[: n - lag]
        valid = denom > 0
        if valid.any():
            C[lag - 1] = (overlap[valid] / denom[valid]).mean()

    return np.arange(1, max_lag + 1), C
