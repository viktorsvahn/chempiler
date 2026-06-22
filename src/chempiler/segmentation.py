"""Coarse trajectory segmentation based on changes in molecular count.

Useful for identifying time windows where the chemical composition is stable
(e.g., between reaction events), which can then be used to compute per-segment
MSDs or other time-averaged properties.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


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


def stitch_identity_tracks(frames, formula, max_gap=50, max_jump=4.0):
    """Chain lifetime segments of *formula* into continuous identity tracks.

    When a molecule transfers its identity to a neighbour via a proton hop
    (Grotthuss mechanism), chempiler records this as one lifetime segment
    ending and a new one starting nearby in space and time.  This function
    detects such handoffs and concatenates the per-segment COM trajectories
    into one unwrapped track per distinct chemical identity.

    Parameters
    ----------
    frames : list of Frame
    formula : str
        Molecular formula to stitch (e.g. ``"HO"`` for hydroxide).
    max_gap : int
        Maximum frame gap between a segment end and the next segment start
        for them to be treated as a single hop event.  Default 50 — in
        typical AIMD water simulations at ~5 fs/frame, the H₃O₂ transition
        state lasts 5–20 frames, so the gap between consecutive HO segments
        can be much larger than 1–2 frames.  Set conservatively above the
        99th percentile of the inter-segment gap distribution for the system
        under study.
    max_jump : float
        Maximum MIC displacement in Å between the last COM of one segment and
        the first COM of the next.  Default 4.0 Å is safely above the O–O
        covalent distance (~2.8 Å) while excluding molecules on the far side
        of the first coordination shell.

    Returns
    -------
    tracks : list of numpy.ndarray, shape (n, 3)
        Unwrapped COM trajectories, one per identity chain.  Each array covers
        all frames in all stitched segments, with spatial offsets applied so
        that consecutive segments are continuous in unwrapped space.
    log : list of dict
        One entry per successful stitch, with keys ``from_seg_end``,
        ``to_seg_start``, ``gap_frames`` and ``jump_ang`` (Å).

    Notes
    -----
    When N molecules of *formula* exist simultaneously (multi-ion systems),
    handoffs at the same transition frame are resolved by minimum-cost
    bipartite matching (Hungarian algorithm) to find the globally optimal
    assignment.
    """
    # Lazy import to avoid circular dependency (msd imports segmentation)
    from .msd import _track_segment

    segs = lifetime_segments(frames, formula)
    if not segs:
        raise ValueError(f"{formula!r} not found in any frame.")

    cell     = np.asarray(frames[-1].atoms.get_cell())
    inv_cell = np.linalg.inv(cell)

    def _mic_vec(a, b):
        """MIC displacement vector from wrapped position a to b."""
        d = b - a
        f = d @ inv_cell
        f -= np.round(f)
        return f @ cell

    # --- Build per-segment data ---
    seg_data = []
    for s, e in segs:
        tracks = _track_segment(frames, formula, s, e)
        if not tracks:
            continue
        track = max(tracks, key=len)

        mols_s = frames[s].formula_to_mols.get(formula, [])
        mols_e = frames[e - 1].formula_to_mols.get(formula, [])
        if not mols_s or not mols_e:
            continue

        seg_data.append({
            'start':         s,
            'end':           e,
            'track':         track,
            'first_wrapped': np.array(frames[s].coms[mols_s[0]]),
            'last_wrapped':  np.array(frames[e - 1].coms[mols_e[0]]),
        })

    if not seg_data:
        return [], []

    seg_data.sort(key=lambda d: d['start'])
    n = len(seg_data)

    # --- Bipartite matching at each transition frame ---
    # Process each unique end-frame once; match all simultaneously-ending
    # segments to simultaneously-starting ones.
    matched_next = [None] * n
    matched_prev = [None] * n

    processed_ends = set()
    for i in range(n):
        et = seg_data[i]['end']
        if et in processed_ends:
            continue
        processed_ends.add(et)

        ending   = [ii for ii in range(n)
                    if seg_data[ii]['end'] == et and matched_next[ii] is None]
        starting = [jj for jj in range(n)
                    if et <= seg_data[jj]['start'] <= et + max_gap
                    and matched_prev[jj] is None]

        if not ending or not starting:
            continue

        # Cost matrix: MIC distance, inf where distance exceeds max_jump
        cost = np.full((len(ending), len(starting)), np.inf)
        for ri, ii in enumerate(ending):
            for ci, jj in enumerate(starting):
                dist = np.linalg.norm(_mic_vec(
                    seg_data[ii]['last_wrapped'],
                    seg_data[jj]['first_wrapped'],
                ))
                if dist <= max_jump:
                    cost[ri, ci] = dist

        if np.all(np.isinf(cost)):
            continue

        # Replace inf with a large finite sentinel so linear_sum_assignment runs
        sentinel  = np.nanmax(cost[np.isfinite(cost)]) * 10
        row_ind, col_ind = linear_sum_assignment(
            np.where(np.isinf(cost), sentinel, cost)
        )
        for ri, ci in zip(row_ind, col_ind):
            if np.isfinite(cost[ri, ci]):
                matched_next[ending[ri]]   = starting[ci]
                matched_prev[starting[ci]] = ending[ri]

    # --- Assemble identity chains ---
    tracks_out = []
    stitch_log = []

    for si in range(n):
        if matched_prev[si] is not None:
            continue   # not a chain head

        pieces = []
        offset = np.zeros(3)
        i = si

        while i is not None:
            d = seg_data[i]
            pieces.append(d['track'] + offset)

            j = matched_next[i]
            if j is not None:
                jump   = _mic_vec(d['last_wrapped'], seg_data[j]['first_wrapped'])
                offset = d['track'][-1] + offset + jump - seg_data[j]['track'][0]
                stitch_log.append({
                    'from_seg_end':  d['end'],
                    'to_seg_start':  seg_data[j]['start'],
                    'gap_frames':    seg_data[j]['start'] - d['end'],
                    'jump_ang':      float(np.linalg.norm(jump)),
                })
            i = j

        if pieces:
            tracks_out.append(np.concatenate(pieces, axis=0))

    if stitch_log:
        jumps = [e['jump_ang'] for e in stitch_log]
        print(
            f"[Chempiler] stitched {len(seg_data)} '{formula}' segments "
            f"→ {len(tracks_out)} identity track(s), "
            f"{len(stitch_log)} hop(s), "
            f"median jump {np.median(jumps):.2f} Å, "
            f"max jump {max(jumps):.2f} Å"
        )

    return tracks_out, stitch_log
