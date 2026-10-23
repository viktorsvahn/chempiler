"""Main entry point for loading and analysing reactive MD trajectories."""

from collections.abc import Sequence
from dataclasses import dataclass

from ase.io import read

from .frame import Frame
from .perception import build_molecules
from .cache import make_cache_key, save_cache, load_cache, cache_n_frames


@dataclass
class HBondMatch:
    """Per-frame nearest linearly hydrogen-bonded acceptor to a donor H.

    Returned by :meth:`ChempilerTrajectory.hbond_partner`. All three
    arrays are the same length (one entry per processed frame); a frame
    with no qualifying acceptor reports NaN / -1 in the corresponding
    slot rather than falling back to "whatever's closest".

    Attributes
    ----------
    distance : numpy.ndarray
        H...acceptor distance (Å), NaN where unmatched.
    angle : numpy.ndarray
        donor_heavy-H...acceptor angle (degrees, 180 = linear), NaN
        where unmatched.
    atom_index : numpy.ndarray of int
        Index of the matched acceptor atom, -1 where unmatched.
    """
    distance: "np.ndarray"
    angle: "np.ndarray"
    atom_index: "np.ndarray"


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


def _make_molecules_whole(pos, cell, molecules, ref_atoms=()):
    """Make every molecule internally whole, in place.

    Each atom is moved to the periodic image nearest its molecule's
    reference atom (the first atom in *ref_atoms* that belongs to the
    molecule, else the molecule's first atom), using ASE's exact
    minimum-image search, so any cell shape is handled.
    """
    import numpy as np
    from ase.geometry import find_mic

    ref_atoms = set(ref_atoms)
    for mol in molecules:
        if len(mol) < 2:
            continue
        ref = next((a for a in mol if a in ref_atoms), mol[0])
        others = np.asarray([a for a in mol if a != ref])
        d, _ = find_mic(pos[others] - pos[ref], cell, pbc=True)
        pos[others] = pos[ref] + d


def _place_molecules_near(pos, cell, molecules, anchor, masses=None):
    """Shift each whole molecule, in place, to the image nearest *anchor*.

    Every molecule is translated rigidly so that its centre of mass
    (mass-weighted if *masses* is given) lies in the periodic image
    nearest *anchor*. Done from scratch each frame -- nothing is carried
    over from earlier frames -- so the placement cannot drift however far
    molecules diffuse over a long trajectory.
    """
    import numpy as np
    from ase.geometry import find_mic

    if not molecules:
        return
    mols = [np.asarray(m) for m in molecules]
    coms = np.array([
        np.average(pos[m], axis=0, weights=None if masses is None else masses[m])
        for m in mols
    ])
    d, _ = find_mic(coms - anchor, cell, pbc=True)
    for m, com, dm in zip(mols, coms, d):
        pos[m] += anchor + dm - com


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

    # Every frame is wrapped around its own anchor from scratch. Carrying
    # positions over from the previous frame instead lets molecules drift
    # into far-away images as they diffuse, so the cluster slowly expands.
    result = []

    for frame in frames:
        pos = frame.atoms.get_positions().copy()
        cell = np.asarray(frame.atoms.get_cell())

        # 1. Make every molecule internally whole.
        _make_molecules_whole(pos, cell, frame.molecules, ref_atoms=center_set)

        # 2-3. Put every molecule in the image nearest this frame's anchor.
        if center_coords is not None:
            anchor = np.asarray(center_coords, dtype=float)
        else:
            anchor = _raw_anchor(pos, frame)
        _place_molecules_near(pos, cell, frame.molecules, anchor)

        # Centre on the anchor's current position in this frame.
        pos -= _centering_shift(pos, frame, np.zeros(3))

        result.append(Atoms(
            symbols=frame.atoms.get_chemical_symbols(),
            positions=pos,
            pbc=False,
        ))

    return result


def _nearest_neighbor_cluster(frame, center_formula, n_neighbors,
                              center_instance=0, target=None):
    """Unwrap, centre, and subset a frame to a species and its N nearest whole molecular neighbours.

    1. Every molecule is made internally whole via the minimum-image
       convention relative to its own first atom (no atom is ever wrapped
       alone).
    2. The mass-weighted centre of mass of the *center_instance*-th molecule
       matching *center_formula* becomes the anchor.
    3. Every other molecule is shifted rigidly so its centre of mass lies
       in the periodic image nearest the anchor. This is redone from
       scratch every frame, so the cluster cannot drift apart as molecules
       diffuse; a molecule only changes image when it is about half a box
       length from the centre, where the nearest image is genuinely
       ambiguous.
    4. Molecules are ranked by closest-approach distance to the centre
       molecule (the minimum distance between any of their atoms and any
       atom of the centre molecule) and the *n_neighbors* closest are kept.
    5. The kept atoms are centred on the anchor and returned as a finite
       ``pbc=False`` cluster with no cell.

    Parameters
    ----------
    frame : Frame
    center_formula : str
        Formula of the centre species, e.g. ``'HO'``.
    n_neighbors : int
        Number of nearest whole molecules to keep around the centre.
    center_instance : int
        Which occurrence of *center_formula* to use if several are present
        in this frame (0-indexed).
    target : str or dict, optional
        Selector (see :func:`chempiler.selectors.resolve`) restricting which
        atoms of candidate molecules count toward the closest-approach
        ranking. ``None`` (default) considers every atom.

    Returns
    -------
    ase.Atoms or None
        ``None`` if *center_formula* is absent from this frame (or has
        fewer than *center_instance* + 1 occurrences).
    """
    import numpy as np
    from ase import Atoms
    from ase.geometry import find_mic
    from .selectors import resolve

    mol_ids = frame.formula_to_mols.get(center_formula, [])
    if not mol_ids or center_instance >= len(mol_ids):
        return None

    atoms = frame.atoms
    syms = atoms.get_chemical_symbols()
    pos = atoms.get_positions().copy()
    masses = atoms.get_masses()
    cell = np.asarray(atoms.get_cell())

    # 1. Make every molecule internally whole.
    _make_molecules_whole(pos, cell, frame.molecules)

    # 2. Anchor = mass-weighted COM of the centre molecule.
    center_mol = mol_ids[center_instance]
    center_idx = np.asarray(frame.molecules[center_mol])
    anchor = np.average(pos[center_idx], axis=0, weights=masses[center_idx])

    # 3. Put every molecule in the image nearest the anchor.
    out_pos = pos
    _place_molecules_near(out_pos, cell, frame.molecules, anchor, masses=masses)

    # 4. Rank other molecules by closest-approach distance to the centre
    #    molecule, using exact minimum-image distances per atom pair rather
    #    than the placed positions: near half a box length a molecule placed
    #    by its centre of mass can have an atom that is closer through
    #    another image, and that atom decides the ranking.
    t_idx = set(resolve(frame, target).tolist()) if target is not None else None
    cand = np.asarray([
        a for mi, mol in enumerate(frame.molecules) if mi != center_mol
        for a in mol if t_idx is None or a in t_idx
    ])

    mol_min_d = {}
    if len(cand):
        rij = (pos[cand][:, None, :] - pos[center_idx][None, :, :]).reshape(-1, 3)
        _, d = find_mic(rij, cell, pbc=True)
        d = d.reshape(len(cand), len(center_idx)).min(axis=1)
        for a, da in zip(frame.atom_to_mol[cand], d):
            mol_min_d[int(a)] = min(mol_min_d.get(int(a), np.inf), float(da))

    nearest = sorted(mol_min_d.items(), key=lambda kv: kv[1])[:n_neighbors]
    mol_set = {center_mol} | {mi for mi, _ in nearest}

    keep = sorted({a for mi in mol_set for a in frame.molecules[mi]})
    cluster = Atoms(
        symbols=[syms[i] for i in keep],
        positions=out_pos[keep] - anchor,
        pbc=False,
    )
    cluster.info['center_formula'] = center_formula
    cluster.info['n_neighbors'] = len(nearest)
    # cluster.info['orig_indices'][j] is this frame's original (full-
    # trajectory) atom index of the cluster's j-th atom -- e.g. atom 3 of
    # the returned cluster is frame.atoms[orig_indices[3]]. Only useful
    # in-memory: plain XYZ (what extract_nearest_neighbors writes) has no
    # metadata support, so this does not survive a save/reload round trip
    # -- see extract_nearest_neighbors' companion JSON for that case.
    cluster.info['orig_indices'] = list(keep)
    return cluster


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


def _attach_companion(frames, path):
    """Attach per-frame calculator data from an HDF5 companion file.

    Per-atom arrays (first dimension == number of atoms) are written to
    ``frame.atoms.arrays``; everything else goes to ``frame.atoms.info``.
    """
    import h5py
    with h5py.File(path, 'r') as hf:
        n_hdf = int(hf.attrs.get('n_frames', 0))
        n = min(len(frames), n_hdf)
        if n < len(frames):
            print(
                f"[Chempiler] companion has {n_hdf} frames, "
                f"trajectory has {len(frames)} — attaching first {n}"
            )
        keys = list(hf.keys())
        for key in keys:
            arr = hf[key][:n]           # shape (n, ...)
            for fi in range(n):
                val = arr[fi]
                frame = frames[fi]
                if val.ndim >= 1 and val.shape[0] == len(frame.atoms):
                    frame.atoms.arrays[key] = val
                elif val.ndim == 0:
                    frame.atoms.info[key] = float(val)
                else:
                    frame.atoms.info[key] = val
    print(
        f"[Chempiler] companion attached → {path} "
        f"({n} frames, {len(keys)} properties: {', '.join(keys)})"
    )


def _apply_persistence(frames, persistence):
    """Remove molecular assignments that appear for fewer than *persistence*
    consecutive frames, replacing them with the surrounding stable assignment.

    Identifies each atom's molecule by the frozenset of all atom indices in
    that molecule. A "blip" is a run of a different frozenset that (a) lasts
    fewer than *persistence* frames and (b) is surrounded on both sides by the
    same frozenset. Blips are reverted to the surrounding value; affected
    frames are rebuilt.

    The most common artifact suppressed is two water molecules briefly merging
    into H4O2 during an O–H stretch that momentarily crosses the bond cutoff.
    """
    if persistence <= 1 or not frames:
        return frames

    from .frame import Frame

    n_frames = len(frames)
    n_atoms  = len(frames[0].atoms)

    mol_membership = []
    for frame in frames:
        membership = [None] * n_atoms
        for mol in frame.molecules:
            fs = frozenset(mol)
            for atom in mol:
                membership[atom] = fs
        mol_membership.append(membership)

    corrected = [list(m) for m in mol_membership]

    for atom_i in range(n_atoms):
        seq = [mol_membership[t][atom_i] for t in range(n_frames)]
        t = 0
        while t < n_frames:
            val     = seq[t]
            run_end = t
            while run_end < n_frames and seq[run_end] == val:
                run_end += 1
            run_len = run_end - t
            if run_len < persistence and t > 0 and run_end < n_frames:
                if seq[t - 1] == seq[run_end]:
                    for tf in range(t, run_end):
                        corrected[tf][atom_i] = seq[t - 1]
            t = run_end

    n_fixed = 0
    for t in range(n_frames):
        if corrected[t] != list(mol_membership[t]):
            mols_dict = {}
            for atom_i, fs in enumerate(corrected[t]):
                if fs is not None:
                    mols_dict.setdefault(fs, []).append(atom_i)
            new_frame = Frame(
                atoms=frames[t].atoms,
                molecules=[sorted(v) for v in mols_dict.values()],
            )
            new_frame.build()
            frames[t] = new_frame
            n_fixed += 1

    if n_fixed:
        print(f"[Chempiler] persistence={persistence}: corrected {n_fixed} frames")
    return frames


class _LazyFrames(Sequence):
    """Frame list read from the HDF5 cache on first use.

    len() is answered from the cache file without loading, so the cache-hit
    paths of rdf(), hbonds_at(), window_counts() and the SDF functions never
    pay for reading the frames.
    """

    def __init__(self, n, loader):
        self._n = n
        self._loader = loader
        self._frames = None

    def _load(self):
        if self._frames is None:
            self._frames = self._loader()
        return self._frames

    @property
    def loaded(self):
        return self._frames is not None

    def __len__(self):
        return self._n if self._frames is None else len(self._frames)

    def __getitem__(self, i):
        return self._load()[i]

    def __iter__(self):
        return iter(self._load())


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

    def build(self, max_frames=None, cache_file=None, persistence=1,
              companion=None):
        """Build (or load from cache) the molecular frame index.

        If *cache_file* exists, frames are loaded from HDF5 lazily: only the
        frame count is read here, and the frames themselves (plus
        persistence filtering and the companion) are loaded the first time
        ``self.frames`` is indexed or iterated. Methods whose own cache_file
        hits therefore never load them. Otherwise the trajectory is parsed,
        molecular topology is detected for each frame, and the result is
        written to *cache_file* for future use.

        Parameters
        ----------
        max_frames : int, optional
            Truncate the trajectory to this many frames.
        cache_file : str, optional
            Path to an HDF5 cache file. Read on hit, written on miss.
        persistence : int
            Minimum number of consecutive frames a molecular assignment must
            persist to be accepted. Assignments that appear for fewer than
            this many frames — surrounded on both sides by the same previous
            assignment — are reverted. Default ``1`` (no filtering). A value
            of ``3`` eliminates most vibrational artefacts (e.g. H4O2 dimers
            from O–H stretch crossings in pure water) without suppressing real
            proton-transfer events that last tens of frames.
        companion : str, optional
            Path to an HDF5 companion file written by
            :class:`~chempiler.CalcResultsWriter`.  Per-atom data (shape
            ``(N,)`` or ``(N, ...)``) is attached to ``frame.atoms.arrays``;
            scalar and global-vector data goes to ``frame.atoms.info``.
            Frame *i* in the companion is matched to ``self.frames[i]``.
        """
        if cache_file:
            try:
                n_frames = cache_n_frames(cache_file)
            except Exception as e:
                print(f"[Chempiler] cache miss → rebuilding ({e})")
            else:
                def _load():
                    frames = load_cache(cache_file)
                    print(f"[Chempiler] loaded cache → {cache_file}")
                    if persistence > 1:
                        _apply_persistence(frames, persistence)
                    if companion is not None:
                        _attach_companion(frames, companion)
                    return frames

                self.frames = _LazyFrames(n_frames, _load)
                print(f"[Chempiler] found cache → {cache_file} "
                      f"({n_frames} frames, loaded on first use)")
                return

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

        if persistence > 1:
            _apply_persistence(self.frames, persistence)

        if cache_file:
            key = make_cache_key(
                self.filename,
                self.mode,
                self.covalent_scale,
                self.coordination_scale,
                max_frames,
                self.sphere_cutoffs,
                persistence,
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

        if companion is not None:
            _attach_companion(self.frames, companion)

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

    def rdf(self, *args, cache_file=None, frame_indices=None, **kwargs):
        """Compute the radial distribution function, optionally cached to disk.

        Delegates to rdf.rdf(); see that function for the full parameter
        documentation (center, target, rmax, dr, integrate, n_workers,
        errors, block_size, return_neff, return_tau).

        Parameters
        ----------
        cache_file : str or Path, optional
            Path to a JSON file. If it exists, ``r``/``g``/``n`` (and
            ``g_err``/``n_err`` with ``errors=True``) are loaded
            from it directly instead of recomputing. If it doesn't, the
            RDF is computed as usual and then saved there -- plain JSON,
            since this is just plot data (a few hundred to a few thousand
            floats) and is worth being able to open and read directly.
            The exact call parameters (center, target, dr, rmax, integrate)
            are stored alongside the arrays and checked on load -- a cache
            computed for a different center/target/dr/rmax/integrate
            raises rather than silently handing back the wrong numbers.
            Use a different *cache_file* per distinct call (e.g. one per
            center/target pair), not one shared file for everything.
        frame_indices : slice, range or iterable of int, optional
            Frames to use. A slice maps onto the usual MD choices:
            ``slice(n_eq, None)`` discards equilibration,
            ``slice(0, n)`` keeps the first *n* frames (convergence with
            simulation length) and ``slice(n_eq, None, 10)`` subsamples
            every 10th frame. Arbitrary indices (e.g. frames in some state)
            are accepted only with ``errors=False``: the error bars need an
            evenly spaced time series. Part of the cache key, so use a
            different *cache_file* per selection. None (default) uses all
            frames.

        Raises
        ------
        RuntimeError
            If any frame is missing its ASE Atoms object (indicates an
            incompatible cache).
        ValueError
            If *cache_file* exists but was computed with different
            parameters than this call, if *frame_indices* is empty or out
            of range, or if it is not evenly spaced and ``errors=True``.
        """
        import inspect
        import numpy as np
        from .rdf import rdf, _result_keys, ERROR_METHOD

        sig = inspect.signature(rdf)
        bound = sig.bind_partial(None, *args, **kwargs)   # frames not needed for the key
        bound.apply_defaults()

        idx, frames_key = self._resolve_frame_indices(frame_indices)
        if idx is not None and bound.arguments['errors'] and not isinstance(frames_key, tuple):
            raise ValueError(
                "errors=True needs evenly spaced frames (a slice or range); "
                "block-averaged error bars are meaningless for an arbitrary "
                "frame selection. Pass errors=False."
            )

        def _frames():
            # Resolved lazily so a cache hit never loads the frames.
            return self.frames if idx is None else [self.frames[i] for i in idx]

        def _check_atoms(frames):
            for i, f in enumerate(frames):
                if f.atoms is None:
                    raise RuntimeError(
                        f"Frame {i} missing ASE atoms. Rebuild the cache."
                    )

        if cache_file is None:
            frames = _frames()
            _check_atoms(frames)
            return rdf(frames, *args, **kwargs)

        import json
        from pathlib import Path

        params = {k: v for k, v in bound.arguments.items() if k not in ('frames', 'n_workers')}
        if not params['return_neff']:
            # Keeps caches written before return_neff existed loadable.
            del params['return_neff']
        if not params['return_tau']:
            del params['return_tau']
        if not params['errors']:
            # Keeps caches written before error bars existed loadable.
            del params['errors'], params['block_size']
        else:
            # Error bars from another estimator (e.g. the earlier
            # autocorrelation-time one) must never load as these.
            params['error_method'] = ERROR_METHOD
        if idx is not None:
            # Only keyed when a strict subset is selected, so full-trajectory
            # caches stay loadable.
            params['frame_indices'] = frames_key
        param_key = repr(sorted(params.items(), key=lambda kv: kv[0]))

        path = Path(cache_file)
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
            if data['param_key'] != param_key:
                raise ValueError(
                    f"{cache_file} was computed with different rdf() "
                    f"parameters or a different error estimator than this "
                    f"call -- delete it to recompute, or use a different "
                    f"cache_file.\n  cached: {data['param_key']}\n  now:    {param_key}"
                )
            print(f"[Chempiler] loaded RDF cache → {cache_file}")
            return tuple(np.array(data[k]) for k in _result_keys(True, True, True, True) if k in data)

        frames = _frames()
        _check_atoms(frames)
        result = rdf(frames, *args, **kwargs)
        keys = _result_keys(params['integrate'], params.get('errors', False),
                            params.get('return_neff', False), params.get('return_tau', False))
        payload = {k: a.tolist() for k, a in zip(keys, result)}
        payload['param_key'] = param_key
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"[Chempiler] saved RDF cache → {cache_file}")
        return result

    def _resolve_frame_indices(self, frame_indices):
        """Normalise a frame selection to an index array and a cache key.

        Returns ``(None, None)`` for no selection or one covering every
        frame in order. An evenly spaced selection is keyed as
        ``(start, stop, step)`` with *stop* exclusive, so equivalent
        spellings (``slice(None, 100)``, ``range(0, 100)``) share a cache;
        any other selection is keyed by a hash of its indices.
        """
        import hashlib
        import numpy as np

        if frame_indices is None:
            return None, None
        n = len(self.frames)
        if isinstance(frame_indices, slice):
            idx = np.arange(*frame_indices.indices(n))
        else:
            idx = np.asarray(list(frame_indices), dtype=np.int64).ravel()
            if ((idx < -n) | (idx >= n)).any():
                raise ValueError(f"frame_indices out of range for {n} frames.")
            idx = idx % n
        if idx.size == 0:
            raise ValueError("frame_indices selects no frames.")

        steps = np.diff(idx)
        if idx.size == 1 or (steps[0] > 0 and (steps == steps[0]).all()):
            step = int(steps[0]) if idx.size > 1 else 1
            key = (int(idx[0]), int(idx[-1]) + step, step)
            if key == (0, n, 1):
                return None, None
            return idx, key
        return idx, hashlib.sha1(idx.tobytes()).hexdigest()[:16]

    def nearest_distances(self, center, target, n_max, stride=1, frame_indices=None):
        """Track the N smallest center-target distances over the trajectory.

        For each frame, resolves *center* to a single atom and *target* to
        a set of candidate atoms, computes MIC-aware distances from the
        centre to every target atom, and keeps the *n_max* smallest.
        *n_max* is purely a cap on how many distances to report, not a
        molecular selection -- contrast with :meth:`extract_nearest_neighbors`,
        which keeps whole molecules; this tracks raw atom-atom distances
        only, e.g. the distance from the OH radical's oxygen to each of
        its 4 nearest water oxygens, frame by frame.

        Parameters
        ----------
        center : str or dict
            Selector (see :func:`chempiler.selectors.resolve`) for the
            single centre atom, e.g. ``{'HO': 'O'}``. Must resolve to
            exactly one atom in every processed frame.
        target : str or dict
            Selector for candidate atoms to measure distance to, e.g.
            ``{'H2O': 'O'}``.
        n_max : int
            Maximum number of nearest distances to keep per frame -- a
            cap, not a requirement. Frames with fewer than *n_max* target
            atoms are padded with NaN in the remaining columns.
        stride : int
            Process every *stride*-th frame when *frame_indices* is None.
        frame_indices : iterable of int, optional
            Explicit frame indices to process. Overrides *stride*.

        Returns
        -------
        numpy.ndarray, shape (n_processed_frames, n_max)
            MIC-aware distances (Å), sorted nearest-to-farthest per row.
            NaN wherever fewer than *n_max* target atoms were present.

        Raises
        ------
        ValueError
            If *center* does not resolve to exactly one atom in some
            processed frame.
        """
        import numpy as np
        from ase.geometry import find_mic
        from .selectors import resolve

        idxs = list(
            frame_indices if frame_indices is not None
            else range(0, len(self.frames), stride)
        )

        out = np.full((len(idxs), n_max), np.nan)
        for row, fi in enumerate(idxs):
            frame = self.frames[fi]
            c_idx = resolve(frame, center)
            if len(c_idx) != 1:
                raise ValueError(
                    f"center selector must resolve to exactly one atom, "
                    f"got {len(c_idx)} in frame {fi}. Use a more specific "
                    f"selector (e.g. a formula+element dict that only "
                    f"matches one atom)."
                )
            t_idx = resolve(frame, target)
            if len(t_idx) == 0:
                continue
            pos = frame.atoms.get_positions()
            cell = np.asarray(frame.atoms.get_cell())
            diff = pos[t_idx] - pos[c_idx[0]]
            _, dists = find_mic(diff, cell, pbc=True)
            dists.sort()
            out[row, :min(n_max, len(dists))] = dists[:n_max]

        return out

    def hbond_partner(self, donor_h, donor_heavy, acceptor,
                      dist_cutoff=2.5, angle_cutoff=150.0,
                      stride=1, frame_indices=None):
        """Track the nearest linearly hydrogen-bonded acceptor to a donor H.

        Unlike :meth:`nearest_distances` (distance only), this applies the
        standard two-part geometric hydrogen-bond definition -- a short
        H...acceptor distance *and* a near-linear donor_heavy-H...acceptor
        angle -- so a nearby but badly-aligned atom doesn't count as a
        match. A frame where nothing satisfies both cutoffs simultaneously
        reports no match (NaN/-1) rather than falling back to "whatever's
        closest": a distance-only search is essentially always "occupied"
        (any O-H in liquid water has *something* nearby), so it's the
        linearity requirement that actually distinguishes a real hydrogen
        bond from incidental proximity.

        Parameters
        ----------
        donor_h : str or dict
            Selector for the single donor hydrogen (must resolve to
            exactly one atom per frame, e.g. a radical's own H).
        donor_heavy : str or dict
            Selector for the heavy atom donor_h is covalently bonded to
            (e.g. that radical's O) -- fixes the donor_heavy-H bond
            vector the angle is measured against.
        acceptor : str or dict
            Selector for candidate acceptor atoms (e.g. water oxygens).
        dist_cutoff : float
            Maximum H...acceptor distance (Å) to qualify as a match.
        angle_cutoff : float
            Minimum donor_heavy-H...acceptor angle (degrees, 180 =
            perfectly linear) to qualify. Candidates failing either
            cutoff are ignored.
        stride : int
            Process every *stride*-th frame when *frame_indices* is None.
        frame_indices : iterable of int, optional
            Explicit frame indices to process. Overrides *stride*.

        Returns
        -------
        HBondMatch
            Among all atoms satisfying both cutoffs, the closest one is
            kept per frame.
        """
        import numpy as np
        from ase.geometry import find_mic
        from .selectors import resolve

        idxs = list(
            frame_indices if frame_indices is not None
            else range(0, len(self.frames), stride)
        )

        distance   = np.full(len(idxs), np.nan)
        angle      = np.full(len(idxs), np.nan)
        atom_index = np.full(len(idxs), -1, dtype=int)

        for row, fi in enumerate(idxs):
            frame = self.frames[fi]
            h_idx = resolve(frame, donor_h)
            d_idx = resolve(frame, donor_heavy)
            a_idx = resolve(frame, acceptor)
            if len(h_idx) != 1 or len(d_idx) != 1 or len(a_idx) == 0:
                continue

            pos  = frame.atoms.get_positions()
            cell = np.asarray(frame.atoms.get_cell())
            hi, di = int(h_idx[0]), int(d_idx[0])

            h_to_acceptor, dist = find_mic(pos[a_idx] - pos[hi], cell, pbc=True)
            # H->donor, not donor->H: for a linear D-H...A bond this points
            # *back* toward D while h_to_acceptor points forward toward A,
            # so the angle between them is ~180 deg (the standard D-H...A
            # convention) rather than ~0 deg.
            h_to_donor, _ = find_mic((pos[di] - pos[hi])[None], cell, pbc=True)

            cos_theta = (h_to_acceptor @ h_to_donor[0]) / (
                dist * np.linalg.norm(h_to_donor[0]) + 1e-12
            )
            theta = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

            qualifies = (dist < dist_cutoff) & (theta > angle_cutoff)
            if not qualifies.any():
                continue

            best = np.where(qualifies)[0][np.argmin(dist[qualifies])]
            distance[row]   = dist[best]
            angle[row]      = theta[best]
            atom_index[row] = a_idx[best]

        return HBondMatch(distance=distance, angle=angle, atom_index=atom_index)

    def window_counts(self, center, target, windows, n_workers=1, block_size=None,
                      return_series=False, cache_file=None):
        """Mean count per centre in distance windows, with error bars.

        Delegates to :func:`chempiler.rdf.window_counts`; see that function
        for the parameters.

        Parameters
        ----------
        cache_file : str or Path, optional
            Path to a JSON file. If it exists, the result is loaded from it
            instead of recomputing; otherwise it is computed and saved there.
            The call parameters, the number of frames and the error
            estimator are stored and checked on load, so a cache from a
            different call, or with error bars from another estimator,
            raises.

        Returns
        -------
        cn, cn_err, cn_neff : numpy.ndarray of shape (n_windows,)
        series : numpy.ndarray of shape (n_frames, n_windows)
            Only with return_series=True.
        """
        from .rdf import window_counts, ERROR_METHOD

        keys = ('cn', 'cn_err', 'cn_neff') + (('series',) if return_series else ())
        if cache_file is None:
            return window_counts(self.frames, center, target, windows,
                                 n_workers=n_workers, block_size=block_size,
                                 return_series=return_series)

        import json
        import numpy as np
        from pathlib import Path

        param_key = repr(sorted(dict(
            center=center, target=target,
            windows=[[float(a), float(b)] for a, b in np.asarray(windows, float).reshape(-1, 2)],
            block_size=block_size, n_frames=len(self.frames),
            error_method=ERROR_METHOD,
            **({'return_series': True} if return_series else {}),
        ).items()))
        path = Path(cache_file)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            if data['param_key'] != param_key:
                raise ValueError(
                    f"{cache_file} was computed with different window_counts() "
                    f"parameters or frames than this call -- delete it, or use "
                    f"a different cache_file.\n  cached: {data['param_key']}\n  now:    {param_key}"
                )
            print(f"[Chempiler] loaded window-count cache → {cache_file}")
            return tuple(np.array(data[k], dtype=float) for k in keys)

        result = window_counts(self.frames, center, target, windows,
                               n_workers=n_workers, block_size=block_size,
                               return_series=return_series)
        # NaN (frames without a centre) is not valid JSON; store it as null.
        payload = {k: np.where(np.isnan(a), None, a).tolist() for k, a in zip(keys, result)}
        payload['param_key'] = param_key
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"[Chempiler] saved window-count cache → {cache_file}")
        return result

    def hbonds_at(self, center, r_HA=2.5, angle_min=150.0, cache_file=None):
        """H-bonds accepted and donated by a chosen atom or molecule, per frame.

        Delegates to :func:`chempiler.hbond.hbonds_at`; see that function
        for the criterion and the forms *center* can take.

        Parameters
        ----------
        cache_file : str or Path, optional
            Path to a JSON file. If it exists, the result is loaded from
            it instead of recomputing; otherwise it is computed and saved
            there. The call parameters (center, r_HA, angle_min) and the
            number of frames are stored and checked on load, so a cache
            from a different call or trajectory raises rather than being
            reused. Use one *cache_file* per distinct call.

        Returns
        -------
        HBondCounts

        Raises
        ------
        ValueError
            If *cache_file* exists but was computed with different
            parameters or a different number of frames.
        """
        from .hbond import hbonds_at, HBondCounts

        if cache_file is None:
            return hbonds_at(self.frames, center, r_HA=r_HA, angle_min=angle_min)

        from pathlib import Path

        param_key = repr(sorted(dict(center=center, r_HA=float(r_HA),
                                     angle_min=float(angle_min),
                                     n_frames=len(self.frames)).items()))
        path = Path(cache_file)
        if path.exists():
            counts, cached_key = HBondCounts.load(path)
            if cached_key != param_key:
                raise ValueError(
                    f"{cache_file} was computed with different hbonds_at() "
                    f"parameters or frames than this call -- delete it, or use "
                    f"a different cache_file.\n  cached: {cached_key}\n  now:    {param_key}"
                )
            print(f"[Chempiler] loaded H-bond cache → {cache_file}")
            return counts

        counts = hbonds_at(self.frames, center, r_HA=r_HA, angle_min=angle_min)
        path.parent.mkdir(parents=True, exist_ok=True)
        counts.save(path, param_key)
        print(f"[Chempiler] saved H-bond cache → {cache_file}")
        return counts

    def nearest_neighbor_angles(self, center, align_to, target, n_max=4,
                                stride=1, frame_indices=None):
        """Track the n_max nearest targets to a centre, and their donor-H...acceptor angle.

        A distance-only, un-opinionated generalisation of
        :meth:`nearest_distances`: alongside each ranked neighbour's
        distance, this also reports the donor_heavy-H...acceptor angle at
        *align_to* (~180 deg = linear, smaller = bent to the side) and the
        neighbour's own atom index. It makes no assumption about how many
        of those neighbours are chemically "special" or what a
        particular distance/angle combination *means* for your system --
        that labelling is analysis-specific (e.g. "the closest one is a
        hemibond", "whichever of the rest is most linear is on-top") and
        belongs in your own analysis code, not in a general trajectory
        method. This just hands you the raw per-neighbour geometry
        (already ranked by distance, so column ``_1`` is always closest)
        to build that labelling from -- see the module docstring example
        or your own analysis notebook for e.g. picking rank 1 as a
        hemibond and the most-linear of the rest as an on-top site.

        Parameters
        ----------
        center : str or dict
            Selector for the single reference atom.
        align_to : str or dict
            Selector for the bond atom fixing the angle's vertex and
            reference direction (e.g. a donor's own H).
        target : str or dict
            Selector for candidate neighbour atoms.
        n_max : int
            Size of the nearest-neighbour pool to report per frame.
        stride : int
            Process every *stride*-th frame when *frame_indices* is None.
        frame_indices : iterable of int, optional
            Explicit frame indices to process. Overrides *stride*.

        Returns
        -------
        pandas.DataFrame
            Indexed by frame, with columns ``dist_1..dist_{n_max}``,
            ``angle_1..angle_{n_max}``, ``atom_1..atom_{n_max}``, ranked
            by distance (rank 1 = closest). Frames with fewer than
            *n_max* candidates are dropped.
        """
        import numpy as np
        import pandas as pd
        from ase.geometry import find_mic
        from .selectors import resolve

        idxs = list(
            frame_indices if frame_indices is not None
            else range(0, len(self.frames), stride)
        )
        rows = []

        for fi in idxs:
            frame = self.frames[fi]
            c_idx = resolve(frame, center)
            a_idx = resolve(frame, align_to)
            t_idx = resolve(frame, target)
            if len(c_idx) != 1 or len(a_idx) == 0 or len(t_idx) < n_max:
                continue

            pos  = frame.atoms.get_positions()
            cell = np.asarray(frame.atoms.get_cell())
            ci   = int(c_idx[0])

            mol_atoms = frame.molecules[int(frame.atom_to_mol[ci])]
            a_in_mol  = [a for a in mol_atoms if a in set(a_idx.tolist())]
            if not a_in_mol:
                continue
            ai = a_in_mol[0]

            c_to_t, c_dist = find_mic(pos[t_idx] - pos[ci], cell, pbc=True)
            pool = np.argsort(c_dist)[:n_max]     # ranked by distance, closest first
            pool_atoms = t_idx[pool]
            pool_dist  = c_dist[pool]

            # donor_heavy-H...acceptor angle for each pool member, vertex
            # at align_to: ~180 deg linear, smaller = bent to the side.
            h_to_acceptor, _ = find_mic(pos[pool_atoms] - pos[ai], cell, pbc=True)
            h_to_donor, _    = find_mic((pos[ci] - pos[ai])[None], cell, pbc=True)
            cos_theta = (h_to_acceptor @ h_to_donor[0]) / (
                np.linalg.norm(h_to_acceptor, axis=1) * np.linalg.norm(h_to_donor[0]) + 1e-12
            )
            pool_angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

            row = {'frame': fi}
            for rank in range(n_max):
                row[f'dist_{rank+1}']  = pool_dist[rank]
                row[f'angle_{rank+1}'] = pool_angle[rank]
                row[f'atom_{rank+1}']  = int(pool_atoms[rank])
            rows.append(row)

        return pd.DataFrame(rows).set_index('frame')

    def msd(self, formula, max_lag=None, correlation_time=None, buffer=5,
            block_size=None, n_blocks=None, dt=None):
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
                   correlation_time=correlation_time, buffer=buffer,
                   block_size=block_size, n_blocks=n_blocks, dt=dt)

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
                               smooth_sigma=0.0,
                               cumulative=False,
                               max_neighbors=None):
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
        cumulative : bool
            If False (default), each cluster contains only molecules in shell i
            (annular).  If True, shells 0..i are all included so that each
            successive inset is a superset of the previous one.
        max_neighbors : int or None
            If set, keep only the *max_neighbors* closest target molecules
            within the selected shell(s).  ``None`` keeps all (default).

        Returns
        -------
        dict of {float: list of ase.Atoms}
            Keys are peak positions (Å, rounded to 4 dp).  Values are up to
            *n_per_peak* local clusters, centred on the center atom, pbc=False.
        """
        import numpy as np
        from ase import Atoms
        from .rdf import rdf_shells

        r, g = np.asarray(rdf[0]), np.asarray(rdf[1])
        shells = rdf_shells(r, g, peak_min_g=peak_min_g, smooth_sigma=smooth_sigma)

        def _molecule_cluster(frame, center_idx, max_shell):
            from .selectors import resolve as _resolve
            pos = frame.atoms.get_positions().copy()
            cell = np.asarray(frame.atoms.get_cell())
            inv_cell = np.linalg.inv(cell)
            syms = frame.atoms.get_chemical_symbols()

            diff = pos - pos[center_idx]
            frac = diff @ inv_cell
            frac -= np.round(frac)
            mic_disp = frac @ cell
            mic_dist = np.linalg.norm(mic_disp, axis=1)

            center_mol = int(frame.atom_to_mol[center_idx])
            target_idx = set(_resolve(frame, target).tolist())

            mol_nearest_d = {}
            for atom_idx in target_idx:
                mol_idx = int(frame.atom_to_mol[atom_idx])
                if mol_idx == center_mol or mol_idx < 0:
                    continue
                d = float(mic_dist[atom_idx])
                if d > float(shells.troughs[-1]):
                    continue
                if mol_idx not in mol_nearest_d or d < mol_nearest_d[mol_idx]:
                    mol_nearest_d[mol_idx] = d

            mol_shells: dict[int, list[int]] = {}
            for mol_idx, d in mol_nearest_d.items():
                idx = int(np.searchsorted(shells.troughs, d, side='right')) - 1
                if 0 <= idx < len(shells.peaks):
                    mol_shells.setdefault(idx, []).append(mol_idx)

            shell_range = range(max_shell + 1) if cumulative else [max_shell]
            candidates = []
            for shell_i in shell_range:
                for mol_idx in mol_shells.get(shell_i, []):
                    candidates.append((mol_nearest_d[mol_idx], mol_idx))
            candidates.sort()
            if max_neighbors is not None:
                candidates = candidates[:max_neighbors]

            mol_set = {center_mol}
            for _, mol_idx in candidates:
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
        for i, rp in enumerate(shells.peaks):
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
                clusters.append(_molecule_cluster(self.frames[fi], h['center_atom'], i))
                if len(clusters) == n_per_peak:
                    break
            if clusters:
                result[float(round(rp, 4))] = clusters

        return result

    def rdf_shell_environments(self, center, target, rdf,
                                n_per_shell=3,
                                peak_min_g=0.5,
                                stride=20,
                                smooth_sigma=0.0,
                                cumulative=False,
                                n=None):
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
        cumulative : bool
            If ``False`` (default) each cluster contains only molecules in the
            annular shell ``[trough_{n-1}, trough_n]`` plus the center molecule
            — good for visualising individual shells without clutter.  If
            ``True`` each cluster contains all molecules from 0 to
            ``trough_n``, i.e. the shell and every inner shell — good for
            storing a complete local environment.
        n : array-like, optional
            Running coordination number from :meth:`rdf` (``integrate=True``).
            When supplied, per-shell and cumulative CNs are attached to the
            returned object as ``envs.shells.cns`` and
            ``envs.shells.cumulative_cns``.

        Returns
        -------
        ShellEnvironments
            A dict subclass keyed by peak position (Å, rounded to 4 dp) —
            drop-in for :func:`~chempiler.rdf.plot_rdf` ``insets``.  Also
            exposes ``.shells`` (:class:`~chempiler.rdf.ShellInfo`) with
            peaks, troughs, and (if *n* supplied) coordination numbers.
        """
        import numpy as np
        from ase import Atoms
        from .selectors import resolve
        from .rdf import rdf_shells, ShellEnvironments

        r, g = np.asarray(rdf[0]), np.asarray(rdf[1])
        shells = rdf_shells(r, g, n=n, peak_min_g=peak_min_g, smooth_sigma=smooth_sigma)

        if len(shells.peaks) == 0:
            return ShellEnvironments({}, shells)

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
            effective_low = 0.0 if cumulative else r_low
            matching = [i for i in t_idx if effective_low <= mic_dist[i] <= r_high]

            mol_set = {int(frame.atom_to_mol[center_idx])}
            for ti in matching:
                mi = int(frame.atom_to_mol[ti])
                if mi >= 0:
                    mol_set.add(mi)

            keep = sorted({a for mi in mol_set for a in frame.molecules[mi]})
            atoms = Atoms(
                symbols=[syms[i] for i in keep],
                positions=mic_disp[keep],
                pbc=False,
            )
            atoms.info['center_atom'] = keep.index(center_idx)
            return atoms

        result = {}
        for i, rp in enumerate(shells.peaks):
            r_low, r_high = float(shells.troughs[i]), float(shells.troughs[i + 1])
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

        return ShellEnvironments(result, shells)

    def extract_nearest_neighbors(self, formula, n_neighbors, output_dir='.',
                                  center_instance=0, target=None,
                                  stride=1, frame_indices=None,
                                  filename=None):
        """Extract a species and its N nearest whole molecular neighbours as a vacuum-cluster XYZ trajectory.

        Frames are processed in order, each building on the last:

        1. Every molecule is made whole via the minimum-image convention
           (no atom is ever wrapped away from the rest of its molecule).
        2. Every atom is placed in the periodic image nearest to where it
           was in the *previous* processed frame -- not nearest to the
           instantaneous anchor -- so a neighbour sitting near the
           periodic boundary does not visually flip to the opposite image
           between frames. The very first processed frame has no
           predecessor, so molecules are instead placed by centre of mass
           relative to the instantaneous anchor.
        3. The mass-weighted centre of mass of the *formula* molecule
           (its *center_instance*-th occurrence) becomes the anchor.
        4. Molecules are ranked by closest-approach distance to the centre
           molecule and the *n_neighbors* closest are kept.
        5. The kept atoms are centred on the anchor and written with
           ``pbc=False`` and no cell -- a finite cluster, frame by frame.

        Parameters
        ----------
        formula : str
            Molecular formula of the centre species (e.g. ``'HO'``).
        n_neighbors : int
            Number of nearest whole molecules to keep around the centre.
        output_dir : str or Path
            Directory in which to write the XYZ file. Created if absent.
        center_instance : int
            Which occurrence of *formula* to use if several are present in a
            frame (0-indexed, default 0).
        target : str or dict, optional
            Selector (see :func:`chempiler.selectors.resolve`) restricting
            which atoms count toward the closest-approach ranking, e.g.
            ``'O'`` to rank neighbours by their oxygen only. ``None``
            (default) considers every atom.
        stride : int
            Process every *stride*-th frame when *frame_indices* is None.
            Frame-to-frame continuity (step 2 above) assumes atoms move
            less than half a box length between two *processed* frames;
            a very large stride (or explicit *frame_indices* far apart)
            can violate that and let a fast-moving neighbour "jump" an
            image after all.
        frame_indices : iterable of int, optional
            Explicit frame indices to process. Overrides *stride*.
        filename : str, optional
            Output file name. Defaults to
            ``'{formula}_nn{n_neighbors}.xyz'``.

        Returns
        -------
        list of ase.Atoms
            The clusters that were written, in frame order. Frames missing
            *formula* are skipped. Each has ``.info['orig_indices']`` --
            cluster atom *j* is ``self.frames[fi].atoms[orig_indices[j]]``
            -- but this is only usable in-memory; it does not survive a
            save/reload of the XYZ file (see the companion JSON below).

        Raises
        ------
        ValueError
            If *formula* is not found in any processed frame.

        Notes
        -----
        Plain XYZ (deliberately used here -- see the ``format='xyz'``
        comment below) carries no metadata, so the mapping from each
        cluster atom back to its original full-trajectory atom index
        would otherwise be lost the moment this is written to disk and
        read back later. To keep it recoverable, a companion JSON file is
        written alongside the XYZ (same name, ``.indices.json`` instead
        of ``.xyz``): a list with one entry per output frame, each
        ``{'orig_frame': int, 'orig_indices': [int, ...]}``, in the same
        order as the frames in the XYZ file. Load both together to
        recover which trajectory atom every cluster atom came from, e.g.::

            frames = ase.io.read('HO_nn5.xyz', ':')
            index_map = json.load(open('HO_nn5.indices.json'))
            # frames[k]'s atom j came from
            # traj.frames[index_map[k]['orig_frame']].atoms[index_map[k]['orig_indices'][j]]
        """
        from pathlib import Path
        from ase.io import write as ase_write
        import json

        idxs = (
            frame_indices if frame_indices is not None
            else range(0, len(self.frames), stride)
        )

        clusters = []
        for fi in idxs:
            cluster = _nearest_neighbor_cluster(
                self.frames[fi], formula, n_neighbors,
                center_instance=center_instance, target=target,
            )
            if cluster is not None:
                cluster.info['frame'] = fi
                clusters.append(cluster)

        if not clusters:
            raise ValueError(f"'{formula}' not found in any processed frame.")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fname = out / (filename or f'{formula}_nn{n_neighbors}.xyz')
        # format='xyz' (not 'extxyz'): a bare coordinate file, no Lattice/pbc metadata.
        ase_write(str(fname), clusters, format='xyz')

        index_map = [
            {'orig_frame': int(c.info['frame']),
             'orig_indices': [int(i) for i in c.info['orig_indices']]}
            for c in clusters
        ]
        index_fname = fname.with_suffix('.indices.json')
        with open(index_fname, 'w') as f:
            json.dump(index_map, f, indent=2)

        print(f'  Written {fname}  ({len(clusters)} frames)')
        print(f'  Written {index_fname}  (original-trajectory index map)')

        return clusters
