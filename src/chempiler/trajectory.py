"""Main entry point for loading and analysing reactive MD trajectories."""

from ase.io import read

from .frame import Frame
from .perception import build_molecules
from .cache import make_cache_key, save_cache, load_cache


def _make_vacuum(frame, center_formula=None, center_instance=0,
                 center_atoms=None):
    """Return a copy of frame.atoms with PBC removed and all molecules made whole.

    Each molecule is reassembled via the minimum-image convention so that
    atoms split across a periodic boundary are translated to be adjacent to
    the first atom of their molecule. The returned Atoms object has pbc=False
    and no cell.

    Centering priority (first match wins):

    1. *center_atoms* — explicit list of atom indices; centroid of those atoms
       is placed at the origin.  Works in every frame regardless of formula.
    2. *center_formula* — formula string; centroid of the *center_instance*-th
       molecule with that formula is placed at the origin.  Frames where the
       formula is absent are left uncentred.
    """
    import numpy as np
    from ase import Atoms

    atoms = frame.atoms
    pos = atoms.get_positions().copy()
    cell = np.asarray(atoms.get_cell())
    inv_cell = np.linalg.inv(cell)

    for mol in frame.molecules:
        if len(mol) < 2:
            continue
        ref = pos[mol[0]]
        for idx in mol[1:]:
            diff = pos[idx] - ref
            frac = diff @ inv_cell
            frac -= np.round(frac)
            pos[idx] = ref + frac @ cell

    if center_atoms is not None:
        centroid = pos[np.asarray(center_atoms)].mean(axis=0)
        pos -= centroid
    elif center_formula is not None:
        mol_ids = frame.formula_to_mols.get(center_formula, [])
        if mol_ids and center_instance < len(mol_ids):
            atom_idx = frame.molecules[mol_ids[center_instance]]
            centroid = pos[atom_idx].mean(axis=0)
            pos -= centroid

    return Atoms(
        symbols=atoms.get_chemical_symbols(),
        positions=pos,
        pbc=False,
    )


def _cluster_around(frames, center_atoms=None, center_formula=None,
                    center_instance=0, center_coords=None):
    """Wrap entire molecules to be near a central entity in each frame.

    For each frame independently:

    1. Make every molecule internally whole (no atom split across a boundary).
    2. Compute the anchor from whichever source is given (priority order:
       *center_atoms* > *center_coords* > *center_formula*).
    3. Translate each whole molecule so its centre of mass is in the nearest
       periodic image of the anchor, using the minimum image convention.
    4. Subtract the anchor so it sits at the origin.

    Because each frame is processed independently, there is no drift and no
    sequential dependency between frames.  Every snapshot shows all molecules
    clustered tightly around the anchor regardless of how far they have
    diffused.

    Parameters
    ----------
    frames : list of Frame
    center_atoms : list of int, optional
        Atom indices whose centroid is the anchor.  Takes precedence over
        *center_coords* and *center_formula*.
    center_coords : array-like of shape (3,), optional
        Cartesian coordinate (Å) to use as a fixed anchor in every frame.
        Takes precedence over *center_formula*.
    center_formula : str, optional
        Formula of the molecule whose centroid is the anchor.
    center_instance : int
        Which occurrence of *center_formula* to use (0-indexed).

    Returns
    -------
    list of ase.Atoms
        Clustered frames with ``pbc=False`` and no cell.
    """
    import numpy as np
    from ase import Atoms

    center_set = set(center_atoms) if center_atoms is not None else set()

    def _make_whole(pos, cell, inv_cell, molecules):
        """Make every molecule internally whole in-place."""
        for mol in molecules:
            if len(mol) < 2:
                continue
            anchor_in = center_set.intersection(mol)
            ref_idx = next(iter(anchor_in)) if anchor_in else mol[0]
            ref = pos[ref_idx]
            for idx in mol:
                if idx == ref_idx:
                    continue
                diff = pos[idx] - ref
                frac = diff @ inv_cell
                frac -= np.round(frac)
                pos[idx] = ref + frac @ cell

    def _raw_anchor(pos, frame):
        """Anchor position from this frame's (post-make_whole) coordinates."""
        if center_atoms is not None:
            return pos[np.asarray(center_atoms)].mean(axis=0)
        if center_formula is not None:
            mol_ids = frame.formula_to_mols.get(center_formula, [])
            if mol_ids and center_instance < len(mol_ids):
                atom_idx = frame.molecules[mol_ids[center_instance]]
                return pos[np.asarray(atom_idx)].mean(axis=0)
        return pos.mean(axis=0)

    def _centering_shift(pos, frame, wrap_ref):
        """Position to subtract so the anchor sits at the origin."""
        if center_atoms is not None:
            # Exact atom position after wrapping
            return pos[np.asarray(center_atoms)].mean(axis=0)
        if center_coords is not None:
            return wrap_ref
        if center_formula is not None:
            mol_ids = frame.formula_to_mols.get(center_formula, [])
            if mol_ids and center_instance < len(mol_ids):
                atom_idx = frame.molecules[mol_ids[center_instance]]
                return pos[np.asarray(atom_idx)].mean(axis=0)
        return wrap_ref

    # For center_coords the wrapping reference is always the fixed user-supplied
    # point.  For center_atoms / center_formula the anchor moves with the
    # molecule; we track it sequentially so the wrapping boundary follows it
    # continuously.  A fixed frame-0 reference would create a boundary that
    # the anchor drifts away from, causing molecules near that boundary to jump
    # by a full lattice vector between frames.
    result = []
    prev_pos = None  # non-centered atom positions from previous frame

    for fi, frame in enumerate(frames):
        pos = frame.atoms.get_positions().copy()
        cell = np.asarray(frame.atoms.get_cell())
        inv_cell = np.linalg.inv(cell)

        # 1. Make every molecule internally whole.
        _make_whole(pos, cell, inv_cell, frame.molecules)

        if fi == 0:
            # Frame 0: determine anchor and wrap every molecule to it so each
            # molecule sits in the image closest to the anchor.
            if center_coords is not None:
                anchor0 = np.asarray(center_coords, dtype=float)
            else:
                anchor0 = _raw_anchor(pos, frame)

            for mol in frame.molecules:
                com = pos[np.asarray(mol)].mean(axis=0)
                frac = (com - anchor0) @ inv_cell
                pos[np.asarray(mol)] += -np.round(frac) @ cell

        elif prev_pos is not None and pos.shape == prev_pos.shape:
            # Frames 1+: move each atom to the nearest image of its position
            # in the previous frame.  This preserves the wrapping chosen in
            # frame 0 and handles any real PBC crossings that occur during the
            # segment without introducing artificial jumps.
            diff = pos - prev_pos
            frac = diff @ inv_cell
            frac -= np.round(frac)
            pos = prev_pos + frac @ cell
        else:
            # Atom count changed (only in synthetic tests / transition buffers):
            # fall back to per-frame molecule wrapping around current anchor.
            anchor_fb = _raw_anchor(pos, frame)
            for mol in frame.molecules:
                com = pos[np.asarray(mol)].mean(axis=0)
                frac = (com - anchor_fb) @ inv_cell
                pos[np.asarray(mol)] += -np.round(frac) @ cell

        # Store non-centred positions for the next iteration.
        prev_pos = pos.copy()

        # Centre on the anchor's current position in this frame.
        pos -= _centering_shift(pos, frame, np.zeros(3))

        result.append(Atoms(
            symbols=frame.atoms.get_chemical_symbols(),
            positions=pos,
            pbc=False,
        ))

    return result


def _recenter(frames, center_atoms=None, center_coords=None, wrap=False):
    """Shift each frame so the chosen entity sits at the box centre, keeping PBC.

    All atom trajectories are MIC-tracked for continuity before the shift is
    applied, so the view is smooth even when the MD code re-wraps atoms.

    Priority: *center_atoms* > *center_coords*.  If neither is given, no shift
    is applied (atoms are still MIC-tracked for continuity).

    Parameters
    ----------
    frames : list of Frame
        Trajectory frames in order.
    center_atoms : list of int, optional
        Atom indices whose centroid is placed at the box centre each frame.
    center_coords : array-like of shape (3,), optional
        Fixed Cartesian point to translate to the box centre.
    wrap : bool
        If ``True``, fold all atom positions back into ``[0, cell)`` after
        translation.  Default ``False``.

    Returns
    -------
    list of ase.Atoms
        Re-centred frames with ``pbc=True`` and the original cell.
    """
    import numpy as np

    frames = list(frames)
    n = len(frames)

    # Pass 1: MIC-track all atom positions and collect refs.
    tracked = []
    refs = []
    prev_pos = None

    for frame in frames:
        raw_pos = frame.atoms.get_positions().copy()
        cell = np.asarray(frame.atoms.get_cell())
        inv_cell = np.linalg.inv(cell)

        if prev_pos is not None and raw_pos.shape == prev_pos.shape:
            diff = raw_pos - prev_pos
            frac = diff @ inv_cell
            frac -= np.round(frac)
            pos = prev_pos + frac @ cell
        else:
            pos = raw_pos

        prev_pos = pos.copy()
        tracked.append(pos)

        if center_atoms is not None:
            refs.append(pos[np.asarray(center_atoms)].mean(axis=0))
        elif center_coords is not None:
            refs.append(np.asarray(center_coords, dtype=float))
        else:
            refs.append(cell.sum(axis=0) / 2)  # no shift

    # Pass 2: apply per-frame centering.
    result = []
    for i, frame in enumerate(frames):
        atoms = frame.atoms.copy()
        cell = np.asarray(atoms.get_cell())
        cell_center = cell.sum(axis=0) / 2
        atoms.set_positions(tracked[i] + (cell_center - refs[i]))
        if wrap:
            atoms.wrap()
        result.append(atoms)

    return result


class ChempilerTrajectory:
    """Load an ASE-readable trajectory and expose analysis methods.

    Molecular topology is rebuilt from scratch for every frame using
    distance-based perception, which correctly handles bond breaking and
    formation in reactive force-field simulations. Results are cached to HDF5
    to avoid redundant work.

    Parameters
    ----------
    filename : str
        Path to an ASE-readable trajectory file (.traj, .xyz, etc.).
    mode : {"molecular", "coordination", "sphere"}
        Perception mode:

        ``"molecular"``
            Covalent bond graph using ASE natural cutoffs scaled by
            *covalent_scale*. Tracks reactive species and bond breaking.

        ``"coordination"``
            Larger cutoffs (scaled by *coordination_scale*) to capture
            first-shell coordination environments via a bond graph.

        ``"sphere"``
            Tight covalent bonds for base molecules (skin=0, excluding
            hydrogen bonds), then explicit distance sphere cutoffs to
            assemble coordination complexes around centre atoms. Gives
            clean, stable species without artefacts from H-bond compression.
            Recommended for MSD and g(r) of solvation complexes.

    covalent_scale : float
        Scale factor applied to ASE natural cutoffs in molecular/coordination
        mode, and to the base-molecule covalent step in sphere mode.
    coordination_scale : float
        Scale factor applied to ASE natural cutoffs in coordination mode.
    sphere_cutoffs : dict or None
        Used only when ``mode='sphere'``. Maps centre–ligand pairs to
        cutoff distances in Å, e.g. ``{'Be-O': 2.4}`` or
        ``{('Be','O'): 2.4}``. If ``None``, cutoffs are auto-detected from
        the first minimum of each pair's distance distribution sampled over
        ~50 trajectory frames.

    Examples
    --------
    >>> traj = ChempilerTrajectory("run.traj")
    >>> traj.build(cache_file="run.h5")
    >>> r, g = traj.rdf(center="O", target="H")

    >>> traj = ChempilerTrajectory("run.traj", mode="sphere")
    >>> traj.build(cache_file="run_sphere.h5")
    >>> traj.summary()   # expect clean BeH8O4 instead of many clusters
    """

    def __init__(
        self,
        filename,
        mode="molecular",
        covalent_scale=1.0,
        coordination_scale=1.3,
        sphere_cutoffs=None,
    ):
        self.filename = filename
        self.mode = mode
        self.covalent_scale = covalent_scale
        self.coordination_scale = coordination_scale
        self.sphere_cutoffs = sphere_cutoffs  # None → auto-detect at build time
        self.frames = []

    def build(self, max_frames=None, cache_file=None):
        """Build (or load from cache) the molecular frame index.

        If *cache_file* exists and was produced with the same parameters,
        frames are loaded from HDF5. Otherwise the trajectory is parsed,
        molecular topology is detected for each frame, and the result is
        written to *cache_file* for future use.

        Parameters
        ----------
        max_frames : int, optional
            Truncate the trajectory to this many frames.
        cache_file : str, optional
            Path to an HDF5 cache file. Read on hit, written on miss.
        """
        if cache_file:
            try:
                self.frames = load_cache(cache_file)
                print(f"[Chempiler] loaded cache → {cache_file}")
                return
            except Exception as e:
                print(f"[Chempiler] cache miss → rebuilding ({e})")

        traj = read(self.filename, index=":")
        if max_frames is not None:
            traj = traj[:max_frames]

        # Resolve sphere cutoffs before the frame loop so auto-detection only
        # runs once and the resolved dict is stored on self for cache keying.
        if self.mode == "sphere":
            from .sphere import (
                parse_cutoffs,
                detect_coordination_centers,
                detect_sphere_cutoffs,
                build_molecules_sphere,
            )
            resolved = parse_cutoffs(self.sphere_cutoffs)
            if not resolved:
                n_sample = min(50, len(traj))
                step = max(1, len(traj) // n_sample)
                sample = traj[::step]
                centers = detect_coordination_centers(sample)
                if centers:
                    resolved = detect_sphere_cutoffs(sample, centers)
                    if not resolved:
                        print(
                            "[Chempiler] sphere mode: centre elements found "
                            "but no sphere cutoffs detected. "
                            "Using tight covalent bonds only."
                        )
                else:
                    print(
                        "[Chempiler] sphere mode: no coordination centres found. "
                        "Using tight covalent bonds only (skin=0.1 Å)."
                    )
            self.sphere_cutoffs = resolved

        self.frames = []
        for atoms in traj:
            if self.mode == "sphere":
                mols = build_molecules_sphere(
                    atoms,
                    self.sphere_cutoffs,
                    bond_scale=self.covalent_scale,
                )
            else:
                mols = build_molecules(
                    atoms,
                    mode=self.mode,
                    bond_scale=self.covalent_scale,
                    coordination_scale=self.coordination_scale,
                )
            frame = Frame(atoms=atoms, molecules=mols)
            frame.build()
            self.frames.append(frame)

        print(f"[Chempiler] built {len(self.frames)} frames")

        if cache_file:
            key = make_cache_key(
                self.filename,
                self.mode,
                self.covalent_scale,
                self.coordination_scale,
                max_frames,
                self.sphere_cutoffs,
            )
            save_cache(
                cache_file,
                self.frames,
                key,
                self.mode,
                self.covalent_scale,
                self.coordination_scale,
            )
            print(f"[Chempiler] cache written → {cache_file}")

    def summary(self):
        """Return a frequency count of molecular formulas across all frames.

        Returns
        -------
        dict
            Maps formula string to total occurrence count summed over frames.
        """
        from collections import Counter
        c = Counter()
        for f in self.frames:
            if f.formulas:
                c.update(f.formulas)
        return dict(c)

    def lifetime_segments(self, formula):
        """Return contiguous frame intervals where *formula* is present.

        Parameters
        ----------
        formula : str
            Molecular formula to search for (e.g. ``"H5O3"``).

        Returns
        -------
        list of tuple of (int, int)
            Half-open intervals ``[start, end)`` of consecutive frames during
            which at least one molecule with *formula* exists.

        Examples
        --------
        >>> segs = traj.lifetime_segments("H5O3")
        >>> traj.frames[segs[0][0]:segs[0][1]]   # frames of first lifetime
        """
        from .segmentation import lifetime_segments
        return lifetime_segments(self.frames, formula)

    def hop_species_distances(self, hop_result, formula="HO", reference="H"):
        """Distance from each hop site to the nearest molecule of a given formula.

        Parameters
        ----------
        hop_result : dict
            Output of ``atom_hop``.
        formula : str
            Formula of the target species (e.g. ``"HO"``).
        reference : {"H", "from", "to"}
            Hop site reference: ``"H"`` (hopping atom), ``"from"`` (donor O),
            or ``"to"`` (acceptor O).

        Returns
        -------
        dict with keys:
            distances, mean, n_measured, n_hops_total
        """
        from .state_engine import hop_species_distances
        return hop_species_distances(self.frames, hop_result, formula=formula, reference=reference)

    def rdf(self, *args, **kwargs):
        """Compute the radial distribution function.

        Delegates to rdf.rdf(); see that function for the full parameter
        documentation.

        Raises
        ------
        RuntimeError
            If any frame is missing its ASE Atoms object (indicates an
            incompatible cache).
        """
        for i, f in enumerate(self.frames):
            if f.atoms is None:
                raise RuntimeError(
                    f"Frame {i} missing ASE atoms. Rebuild the cache."
                )
        from .rdf import rdf
        return rdf(self.frames, *args, **kwargs)

    def msd(self, formula, max_lag=None, correlation_time=None, buffer=5):
        """Compute the mean squared displacement for a molecular species.

        Molecules are tracked within their lifetime segments using minimum-image
        nearest-neighbour COM matching between consecutive frames. Tracks from
        all segments are combined with the standard windowed estimator.

        Parameters
        ----------
        formula : str
            Molecular formula (e.g. ``"H2O"`` or ``"HO"``). Must appear in
            ``traj.summary()``.
        max_lag : int, optional
            Maximum lag in frames. Defaults to half the longest lifetime segment.
        correlation_time : int, optional
            Estimated autocorrelation time of the species motion in frames.
            Segments shorter than ``correlation_time * buffer`` emit a
            :class:`UserWarning` as they may represent spurious intermediates.
        buffer : int
            Safety factor applied to *correlation_time* (default 5).

        Returns
        -------
        lags : numpy.ndarray
            Lag times in frames (1, 2, …, max_lag).
        msd_vals : numpy.ndarray
            Mean squared displacement in Å² at each lag.
        n_samples : numpy.ndarray
            Number of displacement samples at each lag (larger = more reliable).
        """
        from .msd import msd
        return msd(self.frames, formula, max_lag=max_lag,
                   correlation_time=correlation_time, buffer=buffer)

    def extract_segments(self, formula, output_dir='.', vacuum=False,
                         center=False, center_instance=0, center_on=None):
        """Extract trajectory segments containing *formula* and write them as XYZ.

        Each contiguous block of frames in which at least one molecule with
        the given formula is present is written to a separate file named
        ``{formula}_{start}_{end}.xyz``.

        Parameters
        ----------
        formula : str
            Molecular formula to search for (e.g. ``'HO'``, ``'BeH8O4'``).
        output_dir : str or Path
            Directory in which to write the XYZ files. Created if absent.
        vacuum : bool
            If ``True``, remove the periodic cell from each saved frame and
            reassemble all molecules so that atoms split across a periodic
            boundary are translated to be adjacent to the rest of their
            molecule. The output files have no cell and can be visualised as
            free-space clusters. If ``False`` (default), the original
            periodic structure is preserved.
        center : bool
            If ``True`` (requires ``vacuum=True``), translate each frame so
            that the centroid of the *formula* molecule sits at the origin.
            Useful for visualising how the coordination environment evolves
            over time.
        center_instance : int
            Which occurrence of *formula* to centre on when multiple are
            present in a frame (0-indexed, default 0).
        center_on : list of int, optional
            Atom indices whose centroid is placed at the origin in every
            saved frame.  Takes precedence over *center* / *center_instance*
            and implicitly enables *vacuum*.  Use this to lock a specific
            sub-unit (e.g. ``[42, 43]`` for a particular O–H pair inside a
            larger cluster) at the origin rather than the whole-molecule
            centroid.

        Returns
        -------
        list of tuple of (int, int)
            Half-open frame intervals ``[start, end)`` of the extracted
            segments.

        Raises
        ------
        ValueError
            If *formula* does not appear in any frame.
        """
        from pathlib import Path
        from ase.io import write as ase_write
        from .segmentation import lifetime_segments

        segs = lifetime_segments(self.frames, formula)
        if not segs:
            raise ValueError(f"'{formula}' not found in any frame.")

        if center_on is not None:
            vacuum = True

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cf = formula if (vacuum and center and center_on is None) else None
        ca = list(center_on) if center_on is not None else None

        for start, end in segs:
            seg_frames = [self.frames[fi] for fi in range(start, end)]
            if vacuum and (ca is not None or cf is not None):
                frames_out = _cluster_around(
                    seg_frames,
                    center_atoms=ca,
                    center_formula=cf,
                    center_instance=center_instance,
                )
            elif vacuum:
                frames_out = [_make_vacuum(f) for f in seg_frames]
            else:
                frames_out = [f.atoms.copy() for f in seg_frames]

            fname = out / f'{formula}_{start}_{end}.xyz'
            ase_write(str(fname), frames_out, format='extxyz')
            print(f'  Written {fname}  ({end - start} frames)')

        return segs

    def extract_transition(self, formula, segment, buffer=10, event='birth',
                           output_dir='.', vacuum=False, center=False,
                           center_instance=0, center_on=None):
        """Extract the reaction window around a species appearing or disappearing.

        Writes a short XYZ trajectory centred on the creation (``'birth'``)
        or destruction (``'death'``) event of a specific lifetime segment.
        The window spans ``buffer`` frames on each side of the transition
        point, clipped to the trajectory bounds.

        Parameters
        ----------
        formula : str
            Molecular formula (e.g. ``'HO'``, ``'BeH8O4'``).
        segment : int
            0-based index into the list returned by ``lifetime_segments``.
        buffer : int
            Number of frames to include on each side of the event (default 10).
        event : {'birth', 'death', 'both'}
            Which transition to extract. ``'birth'`` captures the appearance
            of the species; ``'death'`` its disappearance; ``'both'`` writes
            a separate file for each.
        output_dir : str or Path
            Directory in which to write the XYZ files. Created if absent.
        vacuum : bool
            Remove PBC and reassemble molecules (same as in
            :meth:`extract_segments`).
        center : bool
            Translate each frame so the *formula* molecule centroid sits at
            the origin (requires ``vacuum=True``). Frames in the buffer region
            where the species is absent are left uncentred.
        center_instance : int
            Which occurrence of *formula* to centre on (0-indexed, default 0).
        center_on : list of int, optional
            Atom indices whose centroid is placed at the origin in every
            saved frame (including buffer frames).  Takes precedence over
            *center* / *center_instance* and implicitly enables *vacuum*.

        Returns
        -------
        tuple of (int, int)
            The ``(start, end)`` of the requested segment.

        Raises
        ------
        ValueError
            If *formula* is not found in any frame.
        IndexError
            If *segment* is out of range.
        """
        from pathlib import Path
        from ase.io import write as ase_write
        from .segmentation import lifetime_segments

        segs = lifetime_segments(self.frames, formula)
        if not segs:
            raise ValueError(f"'{formula}' not found in any frame.")
        if not (-len(segs) <= segment < len(segs)):
            raise IndexError(
                f"Segment index {segment} out of range (0–{len(segs) - 1})."
            )

        if center_on is not None:
            vacuum = True

        n = len(self.frames)
        start, end = segs[segment]
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cf = formula if (vacuum and center and center_on is None) else None
        ca = list(center_on) if center_on is not None else None

        def _write_window(win_start, win_end, label):
            win_frames = [self.frames[fi] for fi in range(win_start, win_end)]
            if vacuum and (ca is not None or cf is not None):
                frames_out = _cluster_around(
                    win_frames,
                    center_atoms=ca,
                    center_formula=cf,
                    center_instance=center_instance,
                )
            elif vacuum:
                frames_out = [_make_vacuum(f) for f in win_frames]
            else:
                frames_out = [f.atoms.copy() for f in win_frames]
            fname = out / f'{formula}_seg{segment}_{label}.xyz'
            ase_write(str(fname), frames_out, format='extxyz')
            print(f'  Written {fname}  ({win_end - win_start} frames, '
                  f'event at frame {start if label == "birth" else end})')

        if event in ('birth', 'both'):
            _write_window(max(0, start - buffer), min(n, start + buffer), 'birth')
        if event in ('death', 'both'):
            _write_window(max(0, end - buffer), min(n, end + buffer), 'death')

        return segs[segment]

    # ── clean data-return API ────────────────────────────────────────────────

    def segment_frames(self, formula, segment):
        """Return raw Frame objects for one lifetime segment.

        Parameters
        ----------
        formula : str
            Molecular formula (e.g. ``'HO'``).
        segment : int
            0-based index into ``lifetime_segments(formula)``.

        Returns
        -------
        list of Frame
        """
        from .segmentation import lifetime_segments
        segs = lifetime_segments(self.frames, formula)
        if not segs:
            raise ValueError(f"'{formula}' not found in any frame.")
        if not (-len(segs) <= segment < len(segs)):
            raise IndexError(f"Segment {segment} out of range (0–{len(segs)-1}).")
        start, end = segs[segment]
        return self.frames[start:end]

    def reaction_window(self, formula, segment, buffer=10, event='birth'):
        """Return frames in the window around a reactive birth or death event.

        Parameters
        ----------
        formula : str
        segment : int
        buffer : int
            Frames to include on each side of the event boundary.
        event : {'birth', 'death'}

        Returns
        -------
        list of Frame
            Raw frames — apply :func:`_recenter` or :func:`_cluster_around`
            before writing to disk.
        int
            Frame index of the event boundary.
        """
        from .segmentation import lifetime_segments
        segs = lifetime_segments(self.frames, formula)
        if not segs:
            raise ValueError(f"'{formula}' not found in any frame.")
        if not (-len(segs) <= segment < len(segs)):
            raise IndexError(f"Segment {segment} out of range (0–{len(segs)-1}).")
        start, end = segs[segment]
        n = len(self.frames)
        if event == 'birth':
            boundary = start
        elif event == 'death':
            boundary = end
        else:
            raise ValueError("event must be 'birth' or 'death'.")
        win_start = max(0, boundary - buffer)
        win_end = min(n, boundary + buffer)
        return self.frames[win_start:win_end], boundary

    def bond_events(self, stride=1):
        """Return bond formation and breaking events across the trajectory.

        Compares the covalent bond set between consecutive sampled frames and
        reports any changes.  Uses the same bond scale as the trajectory's
        molecular perception.

        Parameters
        ----------
        stride : int
            Compare every *stride* frames (default 1 = every frame).

        Returns
        -------
        list of dict
            One entry per frame-pair that has at least one change::

                {
                    'frame':  int,               # index of the later frame
                    'formed': [(i, j, si, sj)],  # new bonds (atom indices + symbols)
                    'broken': [(i, j, si, sj)],  # lost bonds
                }
        """
        from .perception import get_bonds
        events = []
        indices = range(0, len(self.frames), stride)
        prev_bonds = None
        prev_fi = None
        for fi in indices:
            frame = self.frames[fi]
            curr = get_bonds(frame.atoms, bond_scale=self.covalent_scale)
            if prev_bonds is not None:
                formed = curr - prev_bonds
                broken = prev_bonds - curr
                if formed or broken:
                    syms = frame.atoms.get_chemical_symbols()
                    events.append({
                        'frame': fi,
                        'formed': [(i, j, syms[i], syms[j]) for i, j in sorted(formed)],
                        'broken': [(i, j, syms[i], syms[j]) for i, j in sorted(broken)],
                    })
            prev_bonds = curr
            prev_fi = fi
        return events

    def frames_at_distance(self, center, target, r_min, r_max, stride=1):
        """Find frames where any center–target atom pair has distance in [r_min, r_max].

        Useful for extracting representative geometries at a specific RDF peak
        (e.g., the second O–O coordination shell at ~3.5 Å).

        Parameters
        ----------
        center : str or dict
            Selector for the central atoms — same syntax as :meth:`rdf`.
            E.g. ``"O"`` or ``{"HO": "O"}``.
        target : str or dict
            Selector for the target atoms.
        r_min, r_max : float
            Distance window in Å.
        stride : int
            Check every *stride* frames (default 1).

        Returns
        -------
        list of dict
            One entry per matching (frame, atom pair)::

                {
                    'frame':       int,
                    'center_atom': int,
                    'target_atom': int,
                    'distance':    float,
                }

            Pairs are deduplicated when center == target (only i < j reported).
        """
        import numpy as np
        from .selectors import resolve
        hits = []
        for fi in range(0, len(self.frames), stride):
            frame = self.frames[fi]
            atoms = frame.atoms
            pos = atoms.get_positions()
            cell = np.asarray(atoms.get_cell())
            inv_cell = np.linalg.inv(cell)

            c_idx = resolve(frame, center)
            t_idx = resolve(frame, target)
            same = len(c_idx) == len(t_idx) and np.array_equal(c_idx, t_idx)
            if len(c_idx) == 0 or len(t_idx) == 0:
                continue

            pc = pos[c_idx]           # (Nc, 3)
            pt = pos[t_idx]           # (Nt, 3)
            diff = pt[None, :, :] - pc[:, None, :]          # (Nc, Nt, 3)
            frac = diff @ inv_cell
            frac -= np.round(frac)
            dists = np.linalg.norm(frac @ cell, axis=-1)    # (Nc, Nt)

            mask = (dists >= r_min) & (dists <= r_max)
            if same:
                # exclude self-pairs and keep only i < j
                for ci, ti in zip(*np.where(mask)):
                    ai, aj = int(c_idx[ci]), int(t_idx[ti])
                    if ai < aj:
                        hits.append({'frame': fi, 'center_atom': ai,
                                     'target_atom': aj, 'distance': float(dists[ci, ti])})
            else:
                for ci, ti in zip(*np.where(mask)):
                    hits.append({'frame': fi, 'center_atom': int(c_idx[ci]),
                                 'target_atom': int(t_idx[ti]),
                                 'distance': float(dists[ci, ti])})
        return hits

    def rdf_peak_environments(self, center, target, rdf,
                               n_per_peak=3,
                               peak_min_g=0.5, peak_dr=0.12, stride=20,
                               smooth_sigma=0.0):
        """Extract representative local clusters at each peak of a pre-computed RDF.

        Detects peaks in the supplied g(r), then for each peak finds frames
        where a center–target pair sits within ``peak_dr`` Å of that distance
        and extracts the local atomic environment as a vacuum cluster centred
        on the center atom.

        Parameters
        ----------
        center : str
            Element symbol of the central atom (e.g. ``'O'``).
        target : str
            Element symbol of the target atom (e.g. ``'H'``).
        rdf : tuple of (array-like, array-like)
            ``(r, g)`` arrays as returned by :meth:`rdf` — only the first two
            values are used, so ``traj.rdf(...)[:2]`` works.
        n_per_peak : int
            Maximum number of representative clusters to return per peak.
        peak_min_g : float
            Minimum g(r) height to qualify as a peak.
        peak_dr : float
            Half-width of the search window around each detected peak (Å).
        stride : int
            Check every *stride*-th frame when searching for hits.
        smooth_sigma : float
            Standard deviation in Å of a Gaussian applied to g(r) before peak
            detection.  Increase to suppress noise peaks; ``0`` disables
            smoothing (default).

        Returns
        -------
        dict of {float: list of ase.Atoms}
            Keys are peak positions (Å, rounded to 4 dp).  Values are up to
            *n_per_peak* local clusters, centred on the center atom, pbc=False.
        """
        import numpy as np
        from ase import Atoms

        r, g = np.asarray(rdf[0]), np.asarray(rdf[1])

        # Optional Gaussian smoothing to suppress noise before peak detection.
        if smooth_sigma > 0.0:
            dr_step = float(r[1] - r[0])
            sigma_bins = smooth_sigma / dr_step
            half_w = max(1, int(4 * sigma_bins))
            x = np.arange(-half_w, half_w + 1, dtype=float)
            kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
            kernel /= kernel.sum()
            g_det = np.convolve(g, kernel, mode='same')
        else:
            g_det = g

        # Detect local maxima above peak_min_g, skipping r < 0.5 Å.
        valid = r > 0.5
        r_v, g_v = r[valid], g_det[valid]
        is_peak = (
            (g_v[1:-1] > g_v[:-2]) &
            (g_v[1:-1] > g_v[2:]) &
            (g_v[1:-1] > peak_min_g)
        )
        peak_rs = r_v[1:-1][is_peak]

        def _molecule_cluster(frame, center_idx, max_shell):
            from .selectors import resolve as _resolve
            pos = frame.atoms.get_positions().copy()
            cell = np.asarray(frame.atoms.get_cell())
            inv_cell = np.linalg.inv(cell)
            syms = frame.atoms.get_chemical_symbols()
            r_max = float(r[-1])

            diff = pos - pos[center_idx]
            frac = diff @ inv_cell
            frac -= np.round(frac)
            mic_disp = frac @ cell
            mic_dist = np.linalg.norm(mic_disp, axis=1)

            center_mol = int(frame.atom_to_mol[center_idx])
            target_idx = set(_resolve(frame, target).tolist())

            # For each non-center molecule find its nearest target-atom distance.
            mol_nearest_d = {}
            for atom_idx in target_idx:
                mol_idx = int(frame.atom_to_mol[atom_idx])
                if mol_idx == center_mol or mol_idx < 0:
                    continue
                d = float(mic_dist[atom_idx])
                if d > r_max:
                    continue
                if mol_idx not in mol_nearest_d or d < mol_nearest_d[mol_idx]:
                    mol_nearest_d[mol_idx] = d

            # Shell boundaries: midpoint between consecutive peaks.
            # Last shell gets the same half-gap as the previous interval so it
            # has a finite upper bound — molecules beyond it are excluded.
            bounds = np.empty(len(peak_rs) + 1)
            bounds[0] = 0.0
            for k in range(len(peak_rs) - 1):
                bounds[k + 1] = (peak_rs[k] + peak_rs[k + 1]) / 2
            # Last shell has no next peak to mirror against, so extend by a
            # full inter-peak gap (not half) to cover the complete outer shell.
            outer = ((float(peak_rs[-1]) - float(peak_rs[-2])) if len(peak_rs) > 1
                     else float(r[-1]) - float(peak_rs[-1]))
            bounds[-1] = float(peak_rs[-1]) + outer

            mol_shells: dict[int, list[int]] = {}
            for mol_idx, d in mol_nearest_d.items():
                idx = int(np.searchsorted(bounds, d, side='right')) - 1
                if 0 <= idx < len(peak_rs):
                    mol_shells.setdefault(idx, []).append(mol_idx)

            # Non-cumulative: center molecule + only the molecules in this shell.
            mol_set = {center_mol}
            for mol_idx in mol_shells.get(max_shell, []):
                mol_set.add(mol_idx)

            keep = sorted({a for mi in mol_set for a in frame.molecules[mi]})
            atoms = Atoms(
                symbols=[syms[i] for i in keep],
                positions=mic_disp[keep],
                pbc=False,
            )
            atoms.info['center_atom'] = keep.index(center_idx)
            return atoms

        result = {}
        for i, rp in enumerate(peak_rs):
            hits = self.frames_at_distance(
                center, target,
                r_min=rp - peak_dr, r_max=rp + peak_dr,
                stride=stride,
            )
            seen = set()
            clusters = []
            for h in hits:
                fi = h['frame']
                if fi in seen:
                    continue
                seen.add(fi)
                clusters.append(_molecule_cluster(
                    self.frames[fi], h['center_atom'],
                    i,  # max_shell = index of current peak
                ))
                if len(clusters) == n_per_peak:
                    break
            if clusters:
                result[float(round(rp, 4))] = clusters

        return result

    def rdf_shell_environments(self, center, target, rdf,
                                n_per_shell=3,
                                peak_min_g=0.5,
                                stride=20,
                                smooth_sigma=0.0):
        """Extract representative clusters for each trough-delimited coordination shell.

        Shells are defined by the troughs (g(r) minima) between consecutive peaks
        rather than a fixed window around each peak.  For shell N the interval is
        ``[trough_{N-1}, trough_N]``, with the first shell starting at 0 and the
        last ending at the maximum r of the supplied RDF.

        Using troughs as boundaries is more robust than peak-based cutoffs because
        the pair density is at a minimum at a trough, so very few molecules will
        straddle the boundary.  Every whole molecule whose target atom falls inside
        the shell interval is included — no atom is ever cut off at an arbitrary
        distance.

        Parameters
        ----------
        center : str or dict
            Selector for the central atom, same syntax as :meth:`rdf`.
        target : str or dict
            Selector for the target atom, same syntax as :meth:`rdf`.
        rdf : tuple of (array-like, array-like)
            ``(r, g)`` arrays as returned by :meth:`rdf`.
        n_per_shell : int
            Maximum number of representative clusters to return per shell.
        peak_min_g : float
            Minimum g(r) height to qualify as a peak.
        stride : int
            Check every *stride*-th frame when searching for hits.
        smooth_sigma : float
            Standard deviation in Å of a Gaussian applied to g(r) before peak
            and trough detection.  ``0`` disables smoothing (default).

        Returns
        -------
        dict of {float: list of ase.Atoms}
            Keys are peak positions (Å, rounded to 4 dp) — suitable as drop-in
            replacement for :meth:`rdf_peak_environments` when passed to
            :func:`~chempiler.rdf.plot_rdf` as ``insets``.  Values are up to
            *n_per_shell* clusters, each containing all whole molecules in the
            corresponding shell interval, centred on the center atom, pbc=False.
        """
        import numpy as np
        from ase import Atoms
        from .selectors import resolve

        r, g = np.asarray(rdf[0]), np.asarray(rdf[1])

        if smooth_sigma > 0.0:
            dr_step = float(r[1] - r[0])
            sigma_bins = smooth_sigma / dr_step
            half_w = max(1, int(4 * sigma_bins))
            x = np.arange(-half_w, half_w + 1, dtype=float)
            kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
            kernel /= kernel.sum()
            g_det = np.convolve(g, kernel, mode='same')
        else:
            g_det = g

        valid = r > 0.5
        r_v, g_v = r[valid], g_det[valid]
        is_peak = (
            (g_v[1:-1] > g_v[:-2]) &
            (g_v[1:-1] > g_v[2:]) &
            (g_v[1:-1] > peak_min_g)
        )
        peak_rs = r_v[1:-1][is_peak]

        if len(peak_rs) == 0:
            return {}

        # Trough between each consecutive peak pair is the density minimum in
        # that interval — this is where the shell boundary is least likely to
        # bisect a molecule.
        trough_rs = [0.0]
        for i in range(len(peak_rs) - 1):
            r1, r2 = peak_rs[i], peak_rs[i + 1]
            mask = (r >= r1) & (r <= r2)
            if mask.sum() < 3:
                trough_rs.append(float(0.5 * (r1 + r2)))
            else:
                min_idx = int(np.argmin(g_det[mask]))
                trough_rs.append(float(r[mask][min_idx]))
        trough_rs.append(float(r[-1]))

        def _shell_cluster(frame, center_idx, r_low, r_high):
            pos = frame.atoms.get_positions().copy()
            cell = np.asarray(frame.atoms.get_cell())
            inv_cell = np.linalg.inv(cell)
            syms = frame.atoms.get_chemical_symbols()

            diff = pos - pos[center_idx]
            frac = diff @ inv_cell
            frac -= np.round(frac)
            mic_disp = frac @ cell
            mic_dist = np.linalg.norm(mic_disp, axis=1)

            # resolve() handles both plain-string and dict selectors correctly.
            t_idx = set(resolve(frame, target).tolist())
            matching = [i for i in t_idx if r_low <= mic_dist[i] <= r_high]

            mol_set = {int(frame.atom_to_mol[center_idx])}
            for ti in matching:
                mi = int(frame.atom_to_mol[ti])
                if mi >= 0:
                    mol_set.add(mi)

            keep = sorted({a for mi in mol_set for a in frame.molecules[mi]})
            return Atoms(
                symbols=[syms[i] for i in keep],
                positions=mic_disp[keep],
                pbc=False,
            )

        result = {}
        for i, rp in enumerate(peak_rs):
            r_low, r_high = trough_rs[i], trough_rs[i + 1]
            hits = self.frames_at_distance(
                center, target,
                r_min=r_low, r_max=r_high,
                stride=stride,
            )
            seen = set()
            clusters = []
            for h in hits:
                fi = h['frame']
                if fi in seen:
                    continue
                seen.add(fi)
                clusters.append(_shell_cluster(
                    self.frames[fi], h['center_atom'],
                    r_low, r_high,
                ))
                if len(clusters) == n_per_shell:
                    break
            if clusters:
                result[float(round(rp, 4))] = clusters

        return result
