"""Hydrogen bond detection and intermittent autocorrelation."""
from dataclasses import dataclass

import numpy as np
from ase.data import chemical_symbols
from ase.geometry import find_mic

from .selectors import resolve


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
    return _intermittent_acf([_hbonds_frame(f, r_HA) for f in frames], max_lag)


def _intermittent_acf(bond_sets, max_lag):
    """C(τ) = ⟨b(0) b(τ)⟩ / ⟨b(0)⟩ from one set of bond identities per frame."""
    universe = sorted(set().union(*bond_sets))
    if not universe:
        return np.arange(1, max_lag + 1), np.zeros(max_lag)

    idx = {b: i for i, b in enumerate(universe)}
    n, m = len(bond_sets), len(universe)

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


# ---------------------------------------------------------------------------
# H-bonds of a chosen species, any partner
# ---------------------------------------------------------------------------

@dataclass
class HBondCounts:
    """H-bonds accepted and donated by each centre in each frame.

    Returned by :func:`hbonds_at`. One entry per (frame, centre) sample, so
    frames with several centres (e.g. several OH) contribute several
    samples and frames without one contribute none.

    Attributes
    ----------
    frame : numpy.ndarray of int
        Frame index (position in the *frames* passed in) of each sample.
    center : numpy.ndarray of int
        Centre atom index (atom selector) or lowest atom index of the
        centre molecule (formula) of each sample.
    accepted, donated : numpy.ndarray of int
        Number of H-bonds the centre accepts / donates in that frame.
    bonds : list of frozenset
        Per frame, every H-bond involving any centre, as
        ``(donor_heavy, H, acceptor)`` atom-index triplets.
    """
    frame: np.ndarray
    center: np.ndarray
    accepted: np.ndarray
    donated: np.ndarray
    bonds: list

    def distribution(self):
        """Joint probability ``P[n_accepted, n_donated]`` over all samples.

        Returns
        -------
        numpy.ndarray of shape (max accepted + 1, max donated + 1)
            Sums to 1 (all zeros when there are no samples).
        """
        if len(self.accepted) == 0:
            return np.zeros((1, 1))
        P = np.zeros((self.accepted.max() + 1, self.donated.max() + 1))
        np.add.at(P, (self.accepted, self.donated), 1.0)
        return P / P.sum()

    def per_frame(self):
        """Mean accepted and donated per centre in each frame.

        Time series for error analysis, e.g. with
        :func:`~chempiler.core.statistics.estimate_autocorrelation_time`
        and :func:`~chempiler.core.statistics.block_average`.

        Returns
        -------
        accepted, donated : numpy.ndarray of shape (n_frames,)
            NaN in frames without a centre.
        """
        n = len(self.bonds)
        count = np.bincount(self.frame, minlength=n).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            acc = np.bincount(self.frame, weights=self.accepted, minlength=n) / count
            don = np.bincount(self.frame, weights=self.donated, minlength=n) / count
        acc[count == 0] = np.nan
        don[count == 0] = np.nan
        return acc, don

    def acf(self, max_lag=200):
        """Intermittent autocorrelation C(τ) of this species' H-bonds.

        As :func:`hbond_acf`, but only over bonds involving a centre, and
        with the distance + angle criterion used to build this object. A
        bond is identified by its atom triplet, so a proton hop that moves
        the centre onto other atoms ends its bonds.

        Returns
        -------
        lags : numpy.ndarray
        C : numpy.ndarray
        """
        return _intermittent_acf(self.bonds, max_lag)

    def save(self, path, param_key=""):
        """Write to a JSON file; *param_key* is stored for checking on load.

        Each frame's bond set is stored as a sorted list of
        ``[donor_heavy, H, acceptor]`` triplets.
        """
        import json
        payload = {
            "param_key": param_key,
            "frame": self.frame.tolist(),
            "center": self.center.tolist(),
            "accepted": self.accepted.tolist(),
            "donated": self.donated.tolist(),
            "bonds": [sorted(bs) for bs in self.bonds],
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path):
        """Read a file written by :meth:`save`.

        Returns
        -------
        counts : HBondCounts
        param_key : str
        """
        import json
        with open(path) as f:
            d = json.load(f)
        counts = cls(
            frame=np.array(d["frame"], dtype=int),
            center=np.array(d["center"], dtype=int),
            accepted=np.array(d["accepted"], dtype=int),
            donated=np.array(d["donated"], dtype=int),
            bonds=[frozenset(map(tuple, bs)) for bs in d["bonds"]],
        )
        return counts, d["param_key"]


def _centres(frame, center):
    """``[(id, atom indices), ...]`` for each centre in *frame*."""
    if isinstance(center, str) and center not in chemical_symbols:
        return [(int(min(frame.molecules[m])), np.asarray(frame.molecules[m]))
                for m in frame.formula_to_mols.get(center, [])]
    return [(int(a), np.array([a])) for a in resolve(frame, center)]


def _frame_hbonds_at(frame, center, r_HA, angle_min):
    """``(centre id, accepted bonds, donated bonds)`` per centre in one frame."""
    syms = np.asarray(frame.symbols)
    pos = frame.atoms.get_positions()
    cell = frame.atoms.get_cell()
    pbc = frame.atoms.pbc
    mol = np.asarray(frame.atom_to_mol)
    is_h = syms == "H"
    donors = {}

    def donor(h):
        """Heavy atom H is bonded to (nearest in its molecule); None for a free H."""
        if h not in donors:
            heavy = [i for i in frame.molecules[mol[h]] if syms[i] != "H"] if mol[h] >= 0 else []
            if heavy:
                _, r = find_mic(pos[heavy] - pos[h], cell, pbc=pbc)
                donors[h] = int(heavy[int(np.argmin(r))])
            else:
                donors[h] = None
        return donors[h]

    def bonds(h_idx, a_idx):
        found = set()
        if len(h_idx) == 0 or len(a_idx) == 0:
            return found
        v, r = find_mic((pos[a_idx][None, :] - pos[h_idx][:, None]).reshape(-1, 3), cell, pbc=pbc)
        v = v.reshape(len(h_idx), len(a_idx), 3)
        r = r.reshape(len(h_idx), len(a_idx))
        for i, j in zip(*np.nonzero(r < r_HA)):
            h, a = int(h_idx[i]), int(a_idx[j])
            d = donor(h)
            if d is None or d == a:
                continue
            # H->D and H->A: a linear D-H...A bond is ~180 deg between them.
            hd, _ = find_mic((pos[d] - pos[h])[None], cell, pbc=pbc)
            cos = v[i, j] @ hd[0] / (r[i, j] * np.linalg.norm(hd[0]))
            if np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) > angle_min:
                found.add((d, h, a))
        return found

    out = []
    for cid, atoms in _centres(frame, center):
        members = set(atoms.tolist())
        outside = mol != mol[atoms[0]]
        heavy_out = np.flatnonzero(~is_h & outside)
        own_h = np.array([h for h in np.flatnonzero(is_h & ~outside)
                          if h in members or donor(h) in members], dtype=int)
        accepted = bonds(np.flatnonzero(is_h & outside), atoms[~is_h[atoms]])
        donated = bonds(own_h, heavy_out)
        out.append((cid, accepted, donated))
    return out


def hbonds_at(frames, center, r_HA=2.5, angle_min=150.0):
    """H-bonds accepted and donated by a chosen atom or molecule, per frame.

    Geometric criterion: an H-bond D-H...A exists when H...A < *r_HA* and
    the D-H...A angle > *angle_min* (180 = linear), where D is the heavy
    atom H is covalently bonded to (the nearest heavy atom in its
    molecule). The partner can be any atom outside the centre's molecule,
    of any element; free H atoms (no heavy atom in their molecule) never
    donate. Intramolecular H-bonds are not counted.

    Parameters
    ----------
    frames : list of Frame
    center : str or dict
        An atom selector (``{"HO": "O"}``, ``"O"``) gives the H-bonds of
        each selected atom: accepted by it (if it is a heavy atom) and
        donated by the H atoms bonded to it (or by itself, if it is an H).
        A molecular formula (``"HO"``) gives the H-bonds of each whole
        molecule: accepted by any of its heavy atoms, donated by any of
        its H atoms. Every match in a frame is a separate centre.
    r_HA : float
        Maximum H...acceptor distance in Å (default 2.5).
    angle_min : float
        Minimum D-H...A angle in degrees (default 150).

    Returns
    -------
    HBondCounts
        Per-sample counts, with ``.distribution()`` for P(n_acc, n_don),
        ``.per_frame()`` for time series, and ``.acf()`` for lifetimes.
    """
    frame_idx, centre_ids, n_acc, n_don, bond_sets = [], [], [], [], []
    for t, frame in enumerate(frames):
        per_centre = _frame_hbonds_at(frame, center, r_HA, angle_min)
        frame_bonds = set()
        for cid, acc, don in per_centre:
            frame_idx.append(t)
            centre_ids.append(cid)
            n_acc.append(len(acc))
            n_don.append(len(don))
            frame_bonds |= acc | don
        bond_sets.append(frozenset(frame_bonds))

    return HBondCounts(
        frame=np.array(frame_idx, dtype=int),
        center=np.array(centre_ids, dtype=int),
        accepted=np.array(n_acc, dtype=int),
        donated=np.array(n_don, dtype=int),
        bonds=bond_sets,
    )
