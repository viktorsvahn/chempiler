"""Main entry point for loading and analysing reactive MD trajectories."""

from ase.io import read

from .frame import Frame
from .perception import build_molecules
from .cache import make_cache_key, save_cache, load_cache


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
                if not centers:
                    raise RuntimeError(
                        "mode='sphere' auto-detect found no coordination centres. "
                        "Pass sphere_cutoffs explicitly."
                    )
                resolved = detect_sphere_cutoffs(sample, centers)
                if not resolved:
                    raise RuntimeError(
                        "mode='sphere' auto-detect could not determine sphere "
                        "cutoffs from the trajectory sample. "
                        "Pass sphere_cutoffs explicitly."
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
