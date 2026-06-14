"""Main entry point for loading and analysing reactive MD trajectories."""

from ase.io import read

from .frame import Frame
from .perception import build_molecules
from .cache import make_cache_key, save_cache, load_cache


class ChempilerTrajectory:
    """Load an ASE-readable trajectory and expose analysis methods.

    Molecular topology is rebuilt from scratch for every frame using
    distance-based perception (see perception.build_molecules), which
    correctly handles bond breaking and formation in reactive force-field
    simulations. Results are cached to HDF5 to avoid redundant work.

    Parameters
    ----------
    filename : str
        Path to an ASE-readable trajectory file (.traj, .xyz, etc.).
    mode : {"molecular", "coordination"}
        Perception mode: ``"molecular"`` uses covalent radii to detect bonds;
        ``"coordination"`` uses larger radii to capture coordination shells.
    covalent_scale : float
        Scale factor applied to ASE natural cutoffs in molecular mode.
    coordination_scale : float
        Scale factor applied to ASE natural cutoffs in coordination mode.

    Examples
    --------
    >>> traj = ChempilerTrajectory("run.traj")
    >>> traj.build(cache_file="run.h5")
    >>> r, g = traj.rdf(center="O", target="H")
    """

    def __init__(
        self,
        filename,
        mode="molecular",
        covalent_scale=1.0,
        coordination_scale=1.3,
    ):
        self.filename = filename
        self.mode = mode
        self.covalent_scale = covalent_scale
        self.coordination_scale = coordination_scale
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

        self.frames = []
        for atoms in traj:
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
