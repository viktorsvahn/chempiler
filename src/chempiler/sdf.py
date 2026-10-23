"""Spatial distribution functions (2D density maps) around aligned molecules.

For each frame the centre atom is placed at the origin and the vector to the
bond atom is aligned with the +x axis.  Target atom positions in this frame
are accumulated into a 2D histogram, giving a spatial density map of the
solvation structure around the central molecule.

Typical usage::

    hist, xe, ye, meta = spatial_density_map(
        traj.frames,
        center={"HO": "O"}, align_to={"HO": "H"},
        target={"H2O": "H"},
        projection='xz', half_space='+z',
    )
    mol = molecule_overlay('O', 'H', meta['mean_bond_len'], projection='xz')
    fig, ax = plot_sdf(hist, xe, ye, render='both', molecule=mol)
"""

import numpy as np
from scipy.spatial.transform import Rotation
from ase.geometry import find_mic

from .selectors import resolve
from .colours import draw_molecule_2d as _draw_molecule_2d, elem_radius as _elem_radius

_PROJ_AXES = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
_HS_SPEC   = {
    '+x': (0, True),  '-x': (0, False),
    '+y': (1, True),  '-y': (1, False),
    '+z': (2, True),  '-z': (2, False),
}
_ALL_PROJ = set(_PROJ_AXES) | {'cylindrical', 'spherical'}

# ---------------------------------------------------------------------------
# Schoenflies symmetry registry
# ---------------------------------------------------------------------------
# Each entry defines which projections are symmetry-inequivalent, whether a
# secondary alignment atom is needed to fix the rotation around z, and which
# Cartesian directions can be folded (abs) to exploit mirror symmetry.
SCHOENFLIES = {
    'Cinfv': {
        'projections': ('cylindrical', 'spherical'),
        'secondary':   False,
        'bisector':    False,   # single bond → align along that bond
        'folds':       {},
        'description': 'linear, polar — e.g. OH, CO, HCN',
    },
    'Dinfh': {
        'projections': ('cylindrical', 'spherical'),
        'secondary':   False,
        'bisector':    False,   # linear: both bonds collinear, average = same axis
        'folds':       {'z': 'abs'},
        'description': 'linear, centrosymmetric — e.g. CO₂, N₂, O₂',
    },
    'C2v': {
        'projections': ('xz', 'yz', 'cylindrical'),
        'secondary':   True,
        'bisector':    True,    # C2 axis = bisector of H-O-H (or X-A-X angle)
        'folds':       {'x': 'abs', 'y': 'abs'},
        'description': 'bent / pyramidal — e.g. H₂O, SO₂',
    },
    'C3v': {
        'projections': ('xz', 'cylindrical'),
        'secondary':   True,
        'bisector':    True,    # C3 axis = mean of all X-A bonds
        'folds':       {'x': 'abs'},
        'description': 'pyramidal, 3-fold — e.g. NH₃, CHCl₃',
    },
    'Cs': {
        'projections': ('xz', 'yz', 'xy'),
        'secondary':   True,
        'bisector':    False,
        'folds':       {'y': 'abs'},
        'description': 'single mirror plane',
    },
    'C1': {
        'projections': ('xz', 'yz', 'xy'),
        'secondary':   True,
        'bisector':    False,
        'folds':       {},
        'description': 'no symmetry',
    },
    'Td': {
        'projections': ('spherical',),
        'secondary':   True,
        'bisector':    True,    # align along mean of all four X-A bonds
        'folds':       {},
        'description': 'tetrahedral — e.g. CH₄, CCl₄',
    },
    'Oh': {
        'projections': ('spherical',),
        'secondary':   True,
        'bisector':    True,    # align along mean of all six X-A bonds
        'folds':       {},
        'description': 'octahedral — e.g. SF₆',
    },
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _bond_rotation(bond_vec):
    """3×3 rotation matrix that maps *bond_vec* onto the +z axis."""
    bond_len = np.linalg.norm(bond_vec)
    if bond_len < 1e-10:
        return np.eye(3)

    v = bond_vec / bond_len
    target = np.array([0., 0., 1.])
    dot = float(np.clip(v @ target, -1., 1.))

    if abs(dot - 1.) < 1e-10:
        return np.eye(3)
    if abs(dot + 1.) < 1e-10:
        return np.diag([1., -1., -1.])  # 180° around x

    axis = np.cross(v, target)
    axis /= np.linalg.norm(axis)
    return Rotation.from_rotvec(np.arccos(dot) * axis).as_matrix()


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------
def spatial_density_map(
    frames,
    center,
    align_to,
    target,
    projection='xz',
    half_space=None,
    half_space_offset=0.0,
    rmax=5.0,
    bins=150,
    stride=1,
    weights=None,
    exclude_align=True,
    cache_file=None,
):
    """Accumulate a 2D spatial density map of *target* atoms around *center*.

    For every frame, each *center* atom is placed at the origin and the bond
    vector to its *align_to* partner (which must belong to the same molecule)
    is rotated onto +x.  *target* atom positions in this aligned frame are
    binned into a 2D histogram.

    Parameters
    ----------
    frames : list of Frame
    center : str or dict
        Selector for the anchor atom, e.g. ``{"HO": "O"}``.
    align_to : str or dict
        Selector for the atom whose direction defines +x, e.g. ``{"HO": "H"}``.
        Must share a molecule with each *center* atom.
    target : str or dict
        Selector for atoms to accumulate, e.g. ``{"H2O": "H"}``.
    projection : str
        ``'xz'`` / ``'yz'`` / ``'xy'`` — flat Cartesian projection.
        ``'cylindrical'``  — (z, r⊥) map; exploits axial symmetry around the bond.
        ``'spherical'``    — (r, θ) map; θ is the polar angle from the bond axis.
        For axially symmetric molecules (e.g. OH) the cylindrical and spherical
        projections are more statistically efficient than planar ones because all
        atoms at the same (z, r⊥) or (r, θ) contribute regardless of azimuthal φ.
    half_space : str or None
        Restrict to one half of 3D space *before* projecting.
        Only ``'+z'`` and ``'-z'`` are physically meaningful for an axially-aligned
        frame; the other options (``±x``, ``±y``) are provided for completeness.
    half_space_offset : float
        Shift the cutting plane away from the origin (Å).
    rmax : float
        Map half-extent in Å.  For cylindrical/spherical this is the maximum
        radial distance; z spans ``[-rmax, +rmax]``, r⊥ / r span ``[0, rmax]``,
        θ spans ``[0, π]``.
    bins : int
        Number of bins per axis.
    stride : int
        Process every *n*-th frame.
    weights : str or None
        If a string, use ``frame.atoms.arrays[weights]`` as per-atom weights
        for the *target* atoms (e.g. ``weights='spins'`` for a spin-density
        SDF).  Each bin accumulates the *sum of weights* of atoms falling in
        it rather than a count.  The companion HDF5 must be attached first via
        ``traj.build(companion=...)``.
    cache_file : str or Path, optional
        Path to an HDF5 (``.h5``) file. If it exists, ``hist``/``xedges``/
        ``yedges``/``meta`` are loaded from it directly instead of
        recomputing. If it doesn't, this call's result is computed as
        usual and then saved there. The exact call parameters are stored
        alongside the data and checked on load -- a cache computed with a
        different center/align_to/target/projection/rmax/bins/... raises
        rather than silently returning the wrong map.

    Returns
    -------
    hist : ndarray, shape (bins, bins)
        Raw counts (or summed weights when *weights* is set).  Axis convention:
        - planar     — hist[ax0_bin, ax1_bin]
        - cylindrical — hist[z_bin, r_perp_bin]
        - spherical  — hist[r_bin, theta_bin]
    xedges : ndarray
        Bin edges for the *first* histogram axis.
    yedges : ndarray
        Bin edges for the *second* histogram axis.
    meta : dict
        ``'n_center'``, ``'mean_bond_len'``, ``'projection'``, ``'half_space'``.
    """
    param_key = None
    if cache_file is not None:
        import h5py
        import json
        from pathlib import Path

        param_key = repr(sorted({
            'center': center, 'align_to': align_to, 'target': target,
            'projection': projection, 'half_space': half_space,
            'half_space_offset': half_space_offset, 'rmax': rmax,
            'bins': bins, 'stride': stride, 'weights': weights,
            'exclude_align': exclude_align,
        }.items()))
        path = Path(cache_file)
        if path.exists():
            with h5py.File(path, 'r') as hf:
                if hf.attrs['param_key'] != param_key:
                    raise ValueError(
                        f"{cache_file} was computed with different "
                        f"spatial_density_map() parameters than this call "
                        f"-- delete it, or use a different cache_file, if "
                        f"you changed any argument.\n"
                        f"  cached: {hf.attrs['param_key']}\n  now:    {param_key}"
                    )
                hist = hf['hist'][:]
                xedges = hf['xedges'][:]
                yedges = hf['yedges'][:]
                meta = json.loads(hf.attrs['meta_json'])
            print(f"[Chempiler] loaded SDF cache → {cache_file}")
            return hist, xedges, yedges, meta

    if projection not in _ALL_PROJ:
        raise ValueError(f"projection must be one of {sorted(_ALL_PROJ)}")

    # Edge arrays depend on projection type
    if projection in _PROJ_AXES:
        ax0, ax1 = _PROJ_AXES[projection]
        xedges = yedges = np.linspace(-rmax, rmax, bins + 1)
    elif projection == 'cylindrical':
        xedges = np.linspace(-rmax, rmax, bins + 1)   # z (bond)
        yedges = np.linspace(0.0,   rmax, bins + 1)   # r⊥
    else:  # spherical
        xedges = np.linspace(0.0,   rmax, bins + 1)   # r
        yedges = np.linspace(0.0,   np.pi, bins + 1)  # θ (radians)

    hist = np.zeros((bins, bins), dtype=np.float64)
    bond_lens = []
    n_center = 0
    total_target_count = 0
    total_volume = 0.0
    n_density_frames = 0

    for frame in frames[::stride]:
        c_idx = resolve(frame, center)
        a_idx = resolve(frame, align_to)
        t_idx = resolve(frame, target)

        if len(c_idx) == 0 or len(a_idx) == 0 or len(t_idx) == 0:
            continue

        # Bulk (ideal-gas) number density of the target species, as a
        # byproduct of the frames already being iterated -- one target
        # count and one cell volume per frame, independent of how many
        # centre atoms/molecules this frame happens to have. Used as the
        # default plot_sdf(density_reference=...) so a bulk-relative SDF
        # plot doesn't need a separate trajectory load to compute it --
        # see meta['density_reference'] below.
        total_target_count += len(t_idx)
        total_volume += float(frame.atoms.get_volume())
        n_density_frames += 1

        pos  = frame.positions
        cell = np.asarray(frame.atoms.get_cell())
        a_set = set(a_idx.tolist())

        for ci in c_idx:
            mol_id   = int(frame.atom_to_mol[ci])
            mol_atoms = frame.molecules[mol_id]
            a_in_mol  = [a for a in mol_atoms if a in a_set]
            if not a_in_mol:
                continue
            ai = a_in_mol[0]

            # Bond vector via MIC
            bd, bl = find_mic((pos[ai] - pos[ci])[None], cell, pbc=True)
            bond_vec = bd[0]
            bond_lens.append(float(bl[0]))

            R_mat = _bond_rotation(bond_vec)

            # Target positions relative to centre via MIC
            diff = pos[t_idx] - pos[ci]
            mic, _ = find_mic(diff, cell, pbc=True)

            # Remove centre and bond atoms from target set
            excl = (t_idx == ci) | ((t_idx == ai) if exclude_align else False)
            active = ~excl
            mic = mic[active]

            # Per-atom weights for active targets
            w = None
            if weights is not None:
                w = _get_weights(frame, weights, t_idx, active)

            # Rotate
            rot = (R_mat @ mic.T).T

            # Half-space filter
            if half_space is not None:
                hs_ax, hs_pos = _HS_SPEC[half_space]
                cut = float(half_space_offset)
                mask = rot[:, hs_ax] > cut if hs_pos else rot[:, hs_ax] < -cut
                rot = rot[mask]
                if w is not None:
                    w = w[mask]

            # Project and bin
            if projection in _PROJ_AXES:
                h, _, _ = np.histogram2d(
                    rot[:, ax0], rot[:, ax1], bins=[xedges, yedges], weights=w
                )
            elif projection == 'cylindrical':
                r_perp = np.sqrt(rot[:, 0] ** 2 + rot[:, 1] ** 2)
                h, _, _ = np.histogram2d(rot[:, 2], r_perp,
                                         bins=[xedges, yedges], weights=w)
            else:  # spherical
                r_total = np.linalg.norm(rot, axis=1)
                safe_r  = np.where(r_total > 1e-10, r_total, 1.0)
                theta   = np.arccos(np.clip(rot[:, 2] / safe_r, -1.0, 1.0))
                theta[r_total < 1e-10] = 0.0
                h, _, _ = np.histogram2d(r_total, theta,
                                         bins=[xedges, yedges], weights=w)

            hist += h
            n_center += 1

    meta = {
        'n_center':      n_center,
        'mean_bond_len': float(np.mean(bond_lens)) if bond_lens else 0.,
        'projection':    projection,
        'half_space':    half_space,
        'density_reference': (total_target_count / total_volume
                              if total_volume > 0 else None),
    }

    if cache_file is not None:
        import h5py
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, 'w') as hf:
            hf.create_dataset('hist', data=hist, compression='gzip')
            hf.create_dataset('xedges', data=xedges)
            hf.create_dataset('yedges', data=yedges)
            hf.attrs['meta_json'] = json.dumps(meta)
            hf.attrs['param_key'] = param_key
        print(f"[Chempiler] saved SDF cache → {cache_file}")

    return hist, xedges, yedges, meta


def _sym_edges(proj, folds, rmax, bins):
    """Bin edges for one projection, adjusted for any active fold directions."""
    x_abs = folds.get('x') == 'abs'
    y_abs = folds.get('y') == 'abs'
    z_abs = folds.get('z') == 'abs'
    if proj == 'xz':
        return (np.linspace(0 if x_abs else -rmax, rmax, bins + 1),
                np.linspace(0 if z_abs else -rmax, rmax, bins + 1))
    if proj == 'yz':
        return (np.linspace(0 if y_abs else -rmax, rmax, bins + 1),
                np.linspace(0 if z_abs else -rmax, rmax, bins + 1))
    if proj == 'xy':
        return (np.linspace(0 if x_abs else -rmax, rmax, bins + 1),
                np.linspace(0 if y_abs else -rmax, rmax, bins + 1))
    if proj == 'cylindrical':
        return (np.linspace(0 if z_abs else -rmax, rmax, bins + 1),
                np.linspace(0, rmax, bins + 1))
    if proj == 'spherical':
        return (np.linspace(0, rmax, bins + 1),
                np.linspace(0, np.pi / 2 if z_abs else np.pi, bins + 1))
    raise ValueError(f"unknown projection {proj!r}")


def _resolve_cmap(cmap):
    """Return a matplotlib colormap, with a helpful error for unknown names.

    Resolution order:
    1. matplotlib registry (built-ins + anything registered by installed packages)
    2. ``tol.<name>``  — tol_colors registers with this prefix (BuRd → tol.BuRd)
    3. tol_colors.tol_cmap() directly
    4. cmocean, cmasher if installed
    """
    import matplotlib as mpl
    if not isinstance(cmap, str):
        return cmap
    # Direct lookup (covers built-ins + registered colormaps)
    try:
        return mpl.colormaps[cmap]
    except KeyError:
        pass
    # tol_colors registers as 'tol.<name>' — try the prefix automatically
    try:
        return mpl.colormaps[f'tol.{cmap}']
    except KeyError:
        pass
    # tol_colors direct (in case not auto-registered)
    try:
        import tol_colors
        return tol_colors.tol_cmap(cmap)
    except (ImportError, ValueError):
        pass
    # cmocean
    try:
        import cmocean
        return cmocean.cm.cmap_d[cmap]
    except (ImportError, KeyError):
        pass
    # cmasher
    try:
        import cmasher
        return cmasher.cm.__dict__[cmap]
    except (ImportError, KeyError):
        pass
    raise ValueError(
        f"Unknown colormap {cmap!r}. If this is from an optional package, "
        f"install it — e.g.:\n"
        f"    pip install tol-colors   # BuRd, PRGn, sunset, rainbow_PuBr, …\n"
        f"    pip install cmocean\n"
        f"    pip install cmasher"
    ) from None


def _get_weights(frame, key, t_idx, active):
    """Extract per-atom weights for active target indices with a helpful error.

    *key* may be a string (looked up in ``frame.atoms.arrays``) or any callable
    that accepts a frame and returns a 1-D per-atom array, e.g.::

        weights=lambda frame: (
            frame.atoms.arrays['spin_charge_density'][:, 0, 0]
            - frame.atoms.arrays['spin_charge_density'][:, 1, 0]
        )
    """
    if callable(key):
        w = np.asarray(key(frame))
    elif key not in frame.atoms.arrays:
        extra = [k for k in frame.atoms.arrays
                 if k not in ('numbers', 'positions', 'momenta')]
        raise KeyError(
            f"weights={key!r} not found in frame.atoms.arrays "
            f"(non-standard keys present: {extra}). "
            f"Attach a companion HDF5 first: "
            f"traj.build(companion='run_polar.h5')."
        )
    else:
        w = np.asarray(frame.atoms.arrays[key])
    return w[t_idx[active]]


def _bin_one(rot, proj, xe, ye, weights=None):
    """Histogram rotated positions into one projection."""
    if proj in _PROJ_AXES:
        ax0, ax1 = _PROJ_AXES[proj]
        h, _, _ = np.histogram2d(rot[:, ax0], rot[:, ax1],
                                 bins=[xe, ye], weights=weights)
    elif proj == 'cylindrical':
        r_perp = np.sqrt(rot[:, 0] ** 2 + rot[:, 1] ** 2)
        h, _, _ = np.histogram2d(rot[:, 2], r_perp,
                                 bins=[xe, ye], weights=weights)
    else:  # spherical
        r_total = np.linalg.norm(rot, axis=1)
        safe_r  = np.where(r_total > 1e-10, r_total, 1.0)
        theta   = np.arccos(np.clip(rot[:, 2] / safe_r, -1.0, 1.0))
        theta[r_total < 1e-10] = 0.0
        h, _, _ = np.histogram2d(r_total, theta,
                                 bins=[xe, ye], weights=weights)
    return h


def spatial_density_maps(
    frames,
    center,
    align_to,
    target,
    symmetry,
    secondary=None,
    half_space=None,
    half_space_offset=0.0,
    rmax=5.0,
    bins=150,
    stride=1,
    weights=None,
    exclude_align=True,
    bisector=None,
    cache_file=None,
):
    """Accumulate all symmetry-inequivalent SDFs for a Schoenflies point group.

    Parameters
    ----------
    frames, center, align_to, target, half_space, half_space_offset, rmax, bins, stride
        Same as :func:`spatial_density_map`.
    symmetry : str
        Schoenflies symbol — a key of :data:`SCHOENFLIES`.
        Examples: ``'Cinfv'``, ``'Dinfh'``, ``'C2v'``.
    secondary : str or dict or None
        Selector for a second atom in the *center* molecule.  Required when
        ``SCHOENFLIES[symmetry]['secondary']`` is ``True``.  This atom is
        used to fix the remaining rotation around the bond axis (z), placing
        it in the xz half-plane (x > 0).

    Returns
    -------
    hists : dict
        ``{projection: (hist, xedges, yedges)}`` for every projection listed
        in ``SCHOENFLIES[symmetry]['projections']``.  Symmetry folds are
        already applied (e.g. D∞h halves the z range; C2v uses x ≥ 0 in the
        xz panel and y ≥ 0 in the yz panel).
    meta : dict
        ``'n_center'``, ``'mean_bond_len'``, ``'symmetry'``, ``'half_space'``.
    cache_file : str or Path, optional
        Path to an HDF5 (``.h5``) file. If it exists, ``hists``/``meta``
        are loaded from it directly instead of recomputing. If it doesn't,
        this call's result is computed as usual and then saved there. The
        exact call parameters are stored alongside the data and checked
        on load -- a cache computed with different center/align_to/
        target/symmetry/rmax/bins/... raises rather than silently
        returning the wrong maps.
    """
    param_key = None
    if cache_file is not None:
        import h5py
        import json
        from pathlib import Path

        param_key = repr(sorted({
            'center': center, 'align_to': align_to, 'target': target,
            'symmetry': symmetry, 'secondary': secondary,
            'half_space': half_space, 'half_space_offset': half_space_offset,
            'rmax': rmax, 'bins': bins, 'stride': stride,
            'weights': weights, 'exclude_align': exclude_align,
            'bisector': bisector,
        }.items()))
        path = Path(cache_file)
        if path.exists():
            with h5py.File(path, 'r') as hf:
                if hf.attrs['param_key'] != param_key:
                    raise ValueError(
                        f"{cache_file} was computed with different "
                        f"spatial_density_maps() parameters than this call "
                        f"-- delete it, or use a different cache_file, if "
                        f"you changed any argument.\n"
                        f"  cached: {hf.attrs['param_key']}\n  now:    {param_key}"
                    )
                hists = {}
                for proj in hf['hists']:
                    g = hf['hists'][proj]
                    hists[proj] = (g['hist'][:], g['xedges'][:], g['yedges'][:])
                meta = json.loads(hf.attrs['meta_json'])
            print(f"[Chempiler] loaded SDF cache → {cache_file}")
            return hists, meta

    if symmetry not in SCHOENFLIES:
        raise ValueError(
            f"symmetry must be one of {sorted(SCHOENFLIES)}. Got {symmetry!r}."
        )

    sym         = SCHOENFLIES[symmetry]
    projections = sym['projections']
    folds       = sym['folds']
    if bisector is None:
        bisector = sym.get('bisector', False)

    if sym['secondary'] and not bisector and secondary is None:
        raise ValueError(
            f"symmetry={symmetry!r} requires a secondary selector to fix the "
            "rotation around the bond axis (z). Pass secondary=, or use "
            "bisector=True to fix azimuthal alignment via the align_to atoms."
        )

    edges = {p: _sym_edges(p, folds, rmax, bins) for p in projections}
    hists = {p: np.zeros((bins, bins), dtype=np.float64) for p in projections}
    bond_lens = []
    n_center  = 0
    total_target_count = 0
    total_volume = 0.0

    for frame in frames[::stride]:
        c_idx = resolve(frame, center)
        a_idx = resolve(frame, align_to)
        t_idx = resolve(frame, target)
        s_idx = resolve(frame, secondary) if secondary is not None else None

        if len(c_idx) == 0 or len(a_idx) == 0 or len(t_idx) == 0:
            continue

        # Bulk (ideal-gas) number density of the target species -- see the
        # matching comment in spatial_density_map for why this is tracked
        # here rather than requiring a separate trajectory load.
        total_target_count += len(t_idx)
        total_volume += float(frame.atoms.get_volume())

        pos   = frame.positions
        cell  = np.asarray(frame.atoms.get_cell())
        a_set = set(a_idx.tolist())
        s_set = set(s_idx.tolist()) if s_idx is not None else set()

        for ci in c_idx:
            mol_id    = int(frame.atom_to_mol[ci])
            mol_atoms = frame.molecules[mol_id]

            # Primary alignment: bond (or bisector) → +z
            a_in_mol = [a for a in mol_atoms if a in a_set]
            if not a_in_mol:
                continue

            if bisector:
                # Average all align_to vectors → C2 / bisector axis
                diffs, bls = [], []
                for a in a_in_mol:
                    bd_a, bl_a = find_mic((pos[a] - pos[ci])[None], cell, pbc=True)
                    diffs.append(bd_a[0]); bls.append(float(bl_a[0]))
                mean_diff = np.mean(diffs, axis=0)
                bond_lens.append(float(np.mean(bls)))
                R_mat = _bond_rotation(mean_diff)
                ai = None
                # Fix azimuthal: rotate z so first align atom lands in xz (x>0)
                ref_vec = R_mat @ diffs[0]
                phi = np.arctan2(ref_vec[1], ref_vec[0])
                c_, s_ = np.cos(-phi), np.sin(-phi)
                R_mat = np.array([[c_, -s_, 0.], [s_, c_, 0.], [0., 0., 1.]]) @ R_mat
            else:
                ai = a_in_mol[0]
                bd, bl = find_mic((pos[ai] - pos[ci])[None], cell, pbc=True)
                bond_lens.append(float(bl[0]))
                R_mat = _bond_rotation(bd[0])

                # Secondary alignment: rotate around z to bring secondary atom
                # into the xz half-plane (φ → 0).
                if sym['secondary'] and s_set:
                    s_in_mol = [a for a in mol_atoms if a in s_set and a != ai]
                    if s_in_mol:
                        sd, _ = find_mic((pos[s_in_mol[0]] - pos[ci])[None],
                                         cell, pbc=True)
                        sec   = R_mat @ sd[0]
                        phi   = np.arctan2(sec[1], sec[0])
                        c_, s_ = np.cos(-phi), np.sin(-phi)
                        R_z   = np.array([[c_, -s_, 0.], [s_, c_, 0.], [0., 0., 1.]])
                        R_mat = R_z @ R_mat

            # Target positions in aligned frame
            diff   = pos[t_idx] - pos[ci]
            mic, _ = find_mic(diff, cell, pbc=True)
            if bisector:
                excl = (t_idx == ci) | np.isin(t_idx, a_in_mol)
            else:
                excl = (t_idx == ci) | ((t_idx == ai) if exclude_align else False)
            active = ~excl
            rot    = (R_mat @ mic[active].T).T

            # Per-atom weights for active targets
            w = None
            if weights is not None:
                w = _get_weights(frame, weights, t_idx, active)

            # Half-space filter
            if half_space is not None:
                hs_ax, hs_pos = _HS_SPEC[half_space]
                cut  = float(half_space_offset)
                mask = (rot[:, hs_ax] > cut) if hs_pos else (rot[:, hs_ax] < -cut)
                rot  = rot[mask]
                if w is not None:
                    w = w[mask]

            # Apply symmetry folds in-place on a copy
            r = rot.copy()
            for ax, op in folds.items():
                if op == 'abs':
                    idx = {'x': 0, 'y': 1, 'z': 2}[ax]
                    r[:, idx] = np.abs(r[:, idx])

            # Accumulate all projections
            for p in projections:
                xe, ye  = edges[p]
                hists[p] += _bin_one(r, p, xe, ye, weights=w)

            n_center += 1

    meta = {
        'n_center':      n_center,
        'mean_bond_len': float(np.mean(bond_lens)) if bond_lens else 0.,
        'symmetry':      symmetry,
        'half_space':    half_space,
        'density_reference': (total_target_count / total_volume
                              if total_volume > 0 else None),
    }
    result = {p: (hists[p], *edges[p]) for p in projections}

    if cache_file is not None:
        import h5py
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, 'w') as hf:
            grp = hf.create_group('hists')
            for proj, (h, xe, ye) in result.items():
                g = grp.create_group(proj)
                g.create_dataset('hist', data=h, compression='gzip')
                g.create_dataset('xedges', data=xe)
                g.create_dataset('yedges', data=ye)
            hf.attrs['meta_json'] = json.dumps(meta)
            hf.attrs['param_key'] = param_key
        print(f"[Chempiler] saved SDF cache → {cache_file}")

    return result, meta


def sdf_radial_profile(hist, xedges=None, yedges=None, meta=None,
                       n_center=None, density_reference=None):
    """Collapse a spatial density map's ``'spherical'`` projection to g(R), n(R).

    Reading a characteristic distance straight off the peak pixel of a
    bulk-normalised (r⊥, z) or (r, θ) SDF map is unreliable near the
    symmetry axis: ``plot_sdf``'s ``normalize='density'`` divides by a
    volume element that shrinks to zero there (``2π·r⊥·Δr·Δz`` or
    ``2π·r²·sinθ·Δr·Δθ``), so a bin with very few raw counts can still
    show the highest ρ/ρ_bulk ratio on the map even though a much more
    populated (but wider-volume, off-axis) region is the actual dominant
    feature. Collapsing the angular axis of the *raw* histogram counts
    into spherical shells sidesteps that entirely -- each r-shell's count
    is divided by its true ``4/3·π·(r_out³-r_in³)`` volume, giving a
    proper isotropic g(R) with no coordinate-singularity bias, in the
    same ``(r, g, n)`` convention :func:`rdf` returns. Feed the result
    straight into :func:`rdf_shells` to get not just peak positions but
    their trough-bounded distance *ranges* (``ShellInfo.troughs``) --
    including fainter, lower-g features (e.g. a weak second-shell ring)
    that a single peak-position readout would miss.

    Parameters
    ----------
    hist : tuple or array-like
        Either the ``(hists_dict, meta)`` tuple returned by
        :func:`spatial_density_maps` (``hists_dict`` must contain a
        ``'spherical'`` entry -- i.e. the symmetry group passed to
        :func:`spatial_density_maps` must include it, e.g. ``'Cinfv'``),
        or the raw ``(r, θ)`` 2D histogram array from that projection.
    xedges, yedges : array-like, optional
        Bin edges for the raw-array form. Ignored for the tuple form.
    meta : dict, optional
        Overrides the tuple form's own meta (mainly for passing meta
        alongside the raw-array form).
    n_center, density_reference : float, optional
        Override the values tracked in *meta*. *density_reference* is
        required one way or the other -- see :func:`spatial_density_maps`.

    Returns
    -------
    r : numpy.ndarray
        Shell-centre distances (Å).
    g : numpy.ndarray
        ρ(R)/ρ_bulk, isotropic (no axis bias).
    n : numpy.ndarray
        Cumulative coordination number at R.
    """
    _meta = {}
    if (isinstance(hist, tuple) and len(hist) == 2
            and isinstance(hist[0], dict) and xedges is None):
        hists_dict, _meta = hist
        if 'spherical' not in hists_dict:
            raise ValueError(
                "sdf_radial_profile needs the 'spherical' projection -- "
                "call spatial_density_maps(..., symmetry=...) with a "
                "symmetry group whose SCHOENFLIES entry includes it "
                "(e.g. 'Cinfv', 'Dinfh', 'C2v', 'C3v', 'Td', 'Oh')."
            )
        hist, xedges, yedges = hists_dict['spherical']
    elif isinstance(hist, tuple) and len(hist) == 3 and xedges is None:
        hist, xedges, yedges = hist

    if meta is not None:
        _meta = meta

    if n_center is None:
        n_center = _meta.get('n_center', 1)
    if density_reference is None:
        density_reference = _meta.get('density_reference')
    if density_reference is None:
        raise ValueError(
            "density_reference is required -- pass it explicitly, or via "
            "meta from spatial_density_map(s), which tracks it "
            "automatically as it iterates frames."
        )

    hist = np.asarray(hist, dtype=float)
    r_edges = np.asarray(xedges, dtype=float)   # spherical: axis-0 = r
    r  = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = float(r_edges[1] - r_edges[0])
    shell_volume = 4.0 * np.pi * r ** 2 * dr

    shell_counts = hist.sum(axis=1)             # collapse θ -- full-shell counts
    nc = float(max(n_center, 1))

    g = shell_counts / (nc * shell_volume * float(density_reference))
    n = np.cumsum(shell_counts) / nc
    return r, g, n


def sdf_region_membership(frames, center, align_to, target, regions,
                          bisector=False, exclude_align=True, stride=1):
    """Classify *target* atoms by which SDF region they occupy, per frame.

    Rank-based neighbour tracking (e.g. "1st-nearest", "4th-nearest", as
    from :meth:`ChempilerTrajectory.nearest_distances`) is not a stable
    geometric identity: which physical molecule holds a given rank
    changes with the exchange dynamics, and a shell with several
    simultaneous occupants (see the cumulative CN from
    :func:`sdf_radial_profile` + :func:`rdf_shells`) has no single
    "4th-nearest" identity to begin with -- rank 4 is just whichever
    molecule happens to be 4th closest that frame, out of however many
    actually populate the shell. Trying to instead follow one physical
    *atom*'s index across frames doesn't work either, for the opposite
    reason: which atom occupies a given site changes as molecules
    exchange in and out of it.

    This sidesteps both: it re-applies the exact centre/align_to
    alignment :func:`spatial_density_maps` uses to build the SDF (centre
    at the origin, the align_to bond along +z), and classifies every
    target atom, every frame, by which of the supplied geometric
    *regions* -- boxes in (R, θ) relative to that bond axis -- it falls
    into. A region is a place in the aligned frame, not a rank or an
    atom identity, so "is anything in this region right now" survives
    both molecules swapping through it and index churn.

    Parameters
    ----------
    frames : list of Frame
    center : str or dict
        Selector for the single reference atom (must resolve to exactly
        one atom per frame, as with :meth:`nearest_distances`).
    align_to : str or dict
        Selector for the bond atom(s) fixing the alignment axis (e.g.
        the radical's own H). Multiple matches within the centre's
        molecule are averaged when *bisector* is True, else the first
        match is used -- matching :func:`spatial_density_maps`.
    target : str or dict
        Selector for candidate atoms to classify.
    regions : dict of str -> (tuple, tuple or None)
        ``{name: ((r_min, r_max), (theta_min, theta_max))}``, distance in
        Å and theta in degrees from the align_to bond axis (0-180°).
        Use ``theta_range=None`` for "any angle" -- a pure radial shell,
        e.g. straight from :func:`rdf_shells`' ``ShellInfo.troughs``.
        Narrowing theta is what distinguishes a genuine single-partner
        site (e.g. a linear H-bond along the axis) from a broader,
        multiply-occupied ring at a similar R but spread over angle.
    bisector : bool
        Average all align_to matches to fix a C2-type axis instead of
        using a single bond -- see :func:`spatial_density_maps`.
    exclude_align : bool
        Exclude the align_to atom itself from target candidates.
    stride : int

    Returns
    -------
    dict of str -> numpy.ndarray, shape (n_processed_frames,)
        For each region name, the centre-to-occupant distance (Å) of the
        *nearest* target atom found inside that region that frame, or
        NaN if unoccupied. A region built from a singly-occupied shell
        (cumulative CN ≈ 1) will essentially always report 0 or 1
        candidate; for a multiply-occupied region this reports the
        closest one, analogous to ``nearest_distances``' nn1.
    """
    import numpy as np
    from ase.geometry import find_mic
    from .selectors import resolve

    out = {name: [] for name in regions}

    for frame in frames[::stride]:
        c_idx = resolve(frame, center)
        a_idx = resolve(frame, align_to)
        t_idx = resolve(frame, target)
        if len(c_idx) != 1:
            raise ValueError(
                f"center selector must resolve to exactly one atom, got "
                f"{len(c_idx)}. Use a more specific selector."
            )
        if len(a_idx) == 0 or len(t_idx) == 0:
            for name in regions:
                out[name].append(np.nan)
            continue

        pos   = frame.positions
        cell  = np.asarray(frame.atoms.get_cell())
        ci    = int(c_idx[0])
        a_set = set(a_idx.tolist())

        mol_id    = int(frame.atom_to_mol[ci])
        mol_atoms = frame.molecules[mol_id]
        a_in_mol  = [a for a in mol_atoms if a in a_set]
        if not a_in_mol:
            for name in regions:
                out[name].append(np.nan)
            continue

        if bisector:
            diffs = []
            for a in a_in_mol:
                bd_a, _ = find_mic((pos[a] - pos[ci])[None], cell, pbc=True)
                diffs.append(bd_a[0])
            R_mat = _bond_rotation(np.mean(diffs, axis=0))
            ai = None
        else:
            ai = a_in_mol[0]
            bd, _ = find_mic((pos[ai] - pos[ci])[None], cell, pbc=True)
            R_mat = _bond_rotation(bd[0])

        diff   = pos[t_idx] - pos[ci]
        mic, _ = find_mic(diff, cell, pbc=True)
        if bisector:
            excl = (t_idx == ci) | np.isin(t_idx, a_in_mol)
        else:
            excl = (t_idx == ci) | ((t_idx == ai) if exclude_align else False)
        active = ~excl

        dist  = np.linalg.norm(mic[active], axis=1)
        rot_z = (R_mat @ mic[active].T).T[:, 2]
        theta = np.degrees(np.arccos(np.clip(rot_z / np.maximum(dist, 1e-10), -1.0, 1.0)))

        for name, (r_range, theta_range) in regions.items():
            r_lo, r_hi = r_range
            mask = (dist >= r_lo) & (dist < r_hi)
            if theta_range is not None:
                th_lo, th_hi = theta_range
                mask &= (theta >= th_lo) & (theta < th_hi)
            out[name].append(float(dist[mask].min()) if mask.any() else np.nan)

    return {name: np.array(v) for name, v in out.items()}


def molecule_overlay(center_symbol, align_symbol, bond_len, projection='xz',
                     secondary_symbol=None, secondary_bond_len=None,
                     angle_deg=None, extra_atoms=None, bisector=False):
    """Build a molecule dict for use as the *molecule* argument of :func:`plot_sdf`.

    The primary align atom is placed at ``(0, 0, bond_len)`` (+z).  Optional
    secondary and extra atoms are placed in the xz half-plane (x ≥ 0) using the
    supplied bond angle, matching the secondary alignment used by
    :func:`spatial_density_maps`.

    Parameters
    ----------
    center_symbol : str
        Element symbol of the centre atom (e.g. ``'O'``).
    align_symbol : str
        Element symbol of the primary bond atom (e.g. ``'H'``).
    bond_len : float
        Bond length in Å — use ``meta['mean_bond_len']``.
    projection : str
        Must match the projection used when computing the histogram.
    secondary_symbol : str, optional
        Element symbol of a second atom in the centre molecule (e.g. second
        ``'H'`` of H₂O).  Placed in the xz plane using *angle_deg*.
    secondary_bond_len : float, optional
        Bond length for the secondary atom.  Defaults to *bond_len*.
    angle_deg : float, optional
        Angle between the primary bond (+z) and the secondary bond, in degrees
        (e.g. 104.5 for H₂O).  Required when *secondary_symbol* is given.
    extra_atoms : list of (str, array-like), optional
        Additional ``[(symbol, position_3d), ...]`` pairs in the aligned frame
        for full manual control.

    Returns
    -------
    dict
    """
    symbols      = [center_symbol, align_symbol]
    positions_3d = [np.zeros(3), np.array([0., 0., float(bond_len)])]

    if secondary_symbol is not None:
        if angle_deg is None:
            raise ValueError("angle_deg is required when secondary_symbol is given")
        sbl = float(secondary_bond_len if secondary_bond_len is not None else bond_len)
        if bisector:
            # C2 / bisector along +z: both H atoms at ±half_angle, symmetric
            half = np.deg2rad(float(angle_deg) / 2.0)
            # Replace primary H position to be symmetric
            positions_3d[1] = np.array([float(bond_len) * np.sin(half), 0.,
                                         float(bond_len) * np.cos(half)])
            positions_3d.append(np.array([-sbl * np.sin(half), 0., sbl * np.cos(half)]))
        else:
            # H1 along +z, H2 at angle_deg from z in xz plane
            ang  = np.deg2rad(float(angle_deg))
            positions_3d.append(np.array([sbl * np.sin(ang), 0., sbl * np.cos(ang)]))
        symbols.append(secondary_symbol)

    if extra_atoms:
        for sym, pos in extra_atoms:
            symbols.append(sym)
            positions_3d.append(np.asarray(pos, dtype=float))

    return {
        'symbols':      symbols,
        'positions_3d': positions_3d,
        'projection':   projection,
    }


# ---------------------------------------------------------------------------
# Smoothing kernels
# ---------------------------------------------------------------------------
def gto_kernel(alphas, coeffs=None):
    """Return a GTO-style smoothing kernel for use with *smooth_kernel*.

    Each atom contributes Σᵢ cᵢ exp(−αᵢ r²) to the map.  The kernel is
    evaluated on the histogram grid and normalised before convolution, so
    only the *shape* (ratios of αᵢ and cᵢ) matters, not the absolute scale.

    Parameters
    ----------
    alphas : sequence of float
        Gaussian exponents in Å⁻².  Larger values → tighter Gaussians.
    coeffs : sequence of float, optional
        Contraction coefficients.  Defaults to equal weights.

    Returns
    -------
    callable
        ``f(x, y)`` where *x*, *y* are 2-D offset grids in Å.

    Examples
    --------
    Single Gaussian (equivalent to smooth_sigma but via kernel)::

        ax.plot_sdf(data, smooth_kernel=gto_kernel([1.0]))

    STO-3G-like multi-Gaussian (tight core + broad tail)::

        ax.plot_sdf(data, smooth_kernel=gto_kernel(
            alphas=[10.0, 2.0, 0.5],
            coeffs=[0.15,  0.55, 0.44],
        ), smooth_extent=2.0)
    """
    _alphas = np.asarray(alphas, dtype=float)
    _coeffs = (np.ones(len(_alphas)) / len(_alphas)
               if coeffs is None else np.asarray(coeffs, dtype=float))

    def _kernel(x, y):
        r2 = np.asarray(x, dtype=float) ** 2 + np.asarray(y, dtype=float) ** 2
        return sum(float(c) * np.exp(-float(a) * r2)
                   for c, a in zip(_coeffs, _alphas))

    return _kernel


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_sdf(
    hist,
    xedges=None,
    yedges=None,
    ax=None,
    render='fill',
    cmap='inferno',
    levels=12,
    vmax=None,
    vmin=None,
    symmetric=False,
    vcenter=None,
    normalize='density',
    density_reference=None,
    molecule=None,
    mol_scale=1.0,
    orientation='portrait',
    flip_bond=False,
    colorbar=True,
    colorbar_label=None,
    xlabel=None,
    ylabel=None,
    projection=None,
    center_symbol=None,
    align_symbol=None,
    secondary_symbol=None,
    angle_deg=None,
    bisector=False,
    n_center=None,
    smooth_sigma=None,
    smooth_kernel=None,
    smooth_extent=3.0,
    interpolation=None,
    mincount=None,
    grid=False,
    grid_step=None,
    grid_kw=None,
):
    """Plot a 2D spatial density map.

    Parameters
    ----------
    hist : ndarray, shape (N, N)
        Histogram from :func:`spatial_density_map`.
    xedges, yedges : ndarray
        Bin edges from :func:`spatial_density_map`.
    ax : matplotlib Axes or None
        Target axes — a new figure is created when None.
    render : str
        ``'fill'``              — flat colour mesh (pcolormesh).
        ``'contourf'``          — smooth filled contour regions.
        ``'contour'``           — coloured level curves only.
        ``'fill+contour'``      — flat mesh with white level curves.
        ``'contourf+contour'``  — smooth fill with white level curves.
    cmap : str
        Colormap for both fill and level curves (any matplotlib name).
    levels : int or array-like
        If an int (default 12), that many evenly-spaced levels between
        *vmin*/*vmax*. If array-like, used directly as the explicit level
        boundaries (in data units, e.g. counts Å⁻³ after normalisation),
        matching matplotlib's own ``contourf``/``contour`` ``levels``
        convention -- e.g. ``levels=[10, 25, 50, 75]`` for specific
        density thresholds rather than an evenly-spaced count.
    vmax : float or None
        Colour-scale ceiling.  Defaults to the histogram maximum after
        normalisation.
    normalize : str or None
        ``'density'`` — divide by bin area → counts / Å².
        ``'max'``     — scale peak to 1.
        ``None``      — raw counts.
    density_reference : float or None
        Only used when ``normalize='density'``. If given, the local
        number density (counts/Å³) is further divided by this reference,
        giving a dimensionless value where 1.0 means "same as bulk" --
        the same convention g(r) itself uses (g=1 in a featureless
        region, >1 enhanced, <1 depleted), rather than an absolute
        counts/Å³ figure you have to separately remember a bulk value to
        interpret. Pass the ideal-gas/bulk number density of whichever
        element *target* selects, e.g. ``n_O / mean_volume`` for an O
        target -- see :func:`chempiler.selectors.resolve` for computing
        that from your trajectory. ``None`` (default) leaves the density
        as an absolute counts/Å³ value.
    molecule : dict or None
        Central molecule overlay produced by :func:`molecule_overlay`.
    mol_scale : float
        Multiplier applied to covalent radii for the overlay circles.
    orientation : str
        ``'portrait'`` (default) — the bond-related coordinate (z for
        ``xz``/``yz``/``cylindrical``, θ for ``spherical``) runs vertically;
        the perpendicular/magnitude coordinate (x, y, r⊥ or r) runs
        horizontally, increasing left→right.  ``xy`` has no bond-related
        axis, so this just picks x horizontal / y vertical.
        ``'landscape'`` — the same two axes, swapped.
    flip_bond : bool
        Reverse the bond axis direction.  In ``'portrait'`` orientation this
        inverts the y-axis (bond runs top→bottom, placing the align-to atom
        below the centre atom); in ``'landscape'`` it inverts the x-axis
        (bond runs right→left).
    colorbar : bool
        Add a colorbar.
    xlabel, ylabel : str or None
        Axis labels — auto-generated from projection if None.
    mincount : float or None
        Bins with fewer than this many raw (pre-normalisation, pre-smoothing)
        counts are masked out (drawn as transparent) — suppresses noisy,
        poorly-sampled bins the way ``plt.hist2d``'s ``cmin`` does.
    grid : bool
        Overlay a reference grid.  For planar (``xy``/``xz``/``yz``) and
        ``spherical`` projections this is a normal rectilinear grid.  For
        ``cylindrical`` this instead draws concentric semicircles centred on
        the origin (the centre atom) — each ring marks a constant total
        distance ``r = sqrt(z² + r⊥²)`` from the centre, i.e. a spherical
        shell radius, which a rectangular grid cannot show on a (z, r⊥) map.
    grid_step : float or None
        Spacing in Å between gridlines/rings.  Defaults to 1.0 Å for the
        cylindrical rings; ``None`` leaves the rectilinear grid on
        matplotlib's automatic tick spacing.
    grid_kw : dict or None
        Extra styling forwarded to ``ax.grid`` (rectilinear) or to the
        ``Circle`` patches (cylindrical rings).

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    cmap = _resolve_cmap(cmap)  # resolve + validate; substitutes tol.X aliases

    # Allow passing the raw (hists_dict, meta) tuple from spatial_density_maps.
    # projection must be specified to select which entry to plot.
    _meta = {}
    if (isinstance(hist, tuple) and len(hist) == 2
            and isinstance(hist[0], dict) and xedges is None):
        hists_dict, _meta = hist
        proj_key = projection or next(iter(hists_dict))
        hist, xedges, yedges = hists_dict[proj_key]
        if projection is None:
            projection = proj_key

    # Allow passing a (hist, xedges, yedges) 3-tuple directly.
    elif isinstance(hist, tuple) and len(hist) == 3 and xedges is None:
        hist, xedges, yedges = hist

    # Auto-build molecule overlay from meta when symbols are provided.
    if molecule is None and center_symbol and align_symbol:
        bond_len = _meta.get('mean_bond_len')
        if bond_len:
            molecule = molecule_overlay(
                center_symbol, align_symbol, bond_len,
                projection=projection or 'xz',
                secondary_symbol=secondary_symbol,
                angle_deg=angle_deg,
                bisector=bisector,
            )

    # n_center: explicit argument wins, then meta, then 1 (raw counts).
    if n_center is None:
        n_center = _meta.get('n_center', 1)

    # density_reference: explicit argument wins, then meta (the target
    # species' own bulk/ideal-gas density, tracked automatically by
    # spatial_density_map(s) as it iterates frames -- see meta comments
    # there), then None (absolute counts/Å³, no bulk-relative rescaling).
    if density_reference is None:
        density_reference = _meta.get('density_reference')

    # Derive projection early — needed for normalization, labels, and figsize.
    # Explicit projection= wins; fall back to the molecule dict, then 'xz'.
    if projection is not None:
        proj = projection
    elif molecule is not None:
        proj = molecule.get('projection', 'xz')
    else:
        proj = 'xz'

    if orientation not in ('portrait', 'landscape'):
        raise ValueError(f"orientation must be 'portrait' or 'landscape', got {orientation!r}")
    # Which internal axis-swap state gives z (or θ) vertical differs by
    # projection: for cylindrical, z is vertical when the (now internal)
    # swap below is applied; for xy/xz/yz/spherical z (or θ) is already
    # vertical without it.  This keeps 'portrait' meaning the same thing
    # (bond-related coordinate vertical, magnitude coordinate horizontal)
    # for every projection despite the underlying swap direction differing.
    if proj == 'cylindrical':
        bond_vertical = (orientation == 'portrait')
    else:
        bond_vertical = (orientation == 'landscape')

    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])

    # Assign data array and axis edges.
    # Convention: hist first-axis → x-axis when bond_vertical, y-axis otherwise.
    # For non-square projections (cylindrical/spherical) xedges ≠ yedges, so
    # bond_vertical=True must swap which edges go on which axis.
    if bond_vertical:
        data             = hist.copy()
        h_edges, v_edges = yedges, xedges   # swap: first-axis → y, second → x
        hc,      vc      = yc,     xc
    else:
        data             = hist.T.copy()
        h_edges, v_edges = xedges, yedges
        hc,      vc      = xc,     yc

    if ax is None:
        base = 6.0
        width_span  = float(h_edges[-1] - h_edges[0])
        height_span = float(v_edges[-1] - v_edges[0])
        if (proj in _PROJ_AXES or proj == 'cylindrical') and width_span > 0 and height_span > 0:
            # Equal-aspect projections (cylindrical is non-square: e.g.
            # portrait r⊥ span rmax vs z span 2·rmax) — size the figure to
            # the data's actual aspect ratio so there's no wasted whitespace.
            if width_span >= height_span:
                figsize = (base, base * height_span / width_span)
            else:
                figsize = (base * width_span / height_span, base)
        else:
            figsize = (base, base)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Raw-count mask for `mincount` — recorded now, before normalisation
    # rescales the values, and applied after smoothing (below) so the
    # convolution isn't corrupted by NaNs.
    below_mincount = data < mincount if mincount is not None else None

    if normalize == 'density':
        nc = float(max(n_center, 1))
        if proj == 'cylindrical':
            # Volume element: 2π r⊥ Δr Δz  (azimuthal integral absorbed)
            dz = float(xedges[1] - xedges[0])
            dr = float(yedges[1] - yedges[0])
            r_centers = yc   # r⊥ bin centres
            if bond_vertical:
                data = data / (nc * 2 * np.pi * r_centers[np.newaxis, :] * dr * dz)
            else:
                data = data / (nc * 2 * np.pi * r_centers[:, np.newaxis] * dr * dz)
        elif proj == 'spherical':
            # Volume element: 2π r² sin(θ) Δr Δθ
            dr     = float(xedges[1] - xedges[0])
            dtheta = float(yedges[1] - yedges[0])
            r_c    = xc
            theta_c = yc
            sin_t  = np.maximum(np.abs(np.sin(theta_c)), 1e-6)
            if bond_vertical:
                data = data / (nc * 2 * np.pi * r_c[:, np.newaxis] ** 2
                               * sin_t[np.newaxis, :] * dr * dtheta)
            else:
                data = data / (nc * 2 * np.pi * r_c[np.newaxis, :] ** 2
                               * sin_t[:, np.newaxis] * dr * dtheta)
        else:
            dx = float(xedges[1] - xedges[0])
            dy = float(yedges[1] - yedges[0])
            data = data / (nc * dx * dy)
        if density_reference is not None:
            data = data / float(density_reference)
    elif normalize == 'max' and data.max() > 0:
        data = data / data.max()

    # Optional smoothing — applied BEFORE colour scale so vmin/vmax reflect
    # the actual smoothed range (important for sparse/spin-density data where
    # peak heights drop significantly after convolution).
    #
    # Axis note: data is stored with v-edges along axis-0 and h-edges along
    # axis-1 (hist.T layout for bond_vertical=False, or hist for True but with
    # h/v swapped).  Sigma and kernel extent must be keyed to the CORRECT axis:
    #   axis-0 bin size = dy (v-edges),  axis-1 bin size = dx (h-edges).
    #
    # Cylindrical boundary: r⊥ ≥ 0 always; reflect mode at the r⊥ boundary
    # avoids undersmoothing near the axis (constant/zero padding is wrong).
    dx = float(h_edges[1] - h_edges[0])   # axis-1 bin size (h direction)
    dy = float(v_edges[1] - v_edges[0])   # axis-0 bin size (v direction)

    if smooth_sigma is not None and smooth_kernel is not None:
        raise ValueError("Specify smooth_sigma or smooth_kernel, not both.")

    if smooth_sigma is not None or smooth_kernel is not None:
        # Per-axis boundary mode: reflect on r⊥ axis for cylindrical projection
        if proj == 'cylindrical':
            # For bond_vertical=False: axis-0 = r⊥, axis-1 = z
            # For bond_vertical=True:  axis-0 = z,   axis-1 = r⊥
            if bond_vertical:
                smooth_mode = ['constant', 'reflect']
            else:
                smooth_mode = ['reflect', 'constant']
        else:
            smooth_mode = 'constant'

        if smooth_sigma is not None:
            from scipy.ndimage import gaussian_filter
            # axis-0 uses dy, axis-1 uses dx (NOT the other way round)
            data = gaussian_filter(
                data,
                sigma=[smooth_sigma / dy, smooth_sigma / dx],
                mode=smooth_mode,
            )
        else:
            from scipy.ndimage import convolve
            if callable(smooth_kernel):
                # Build kernel grid matching data's axis ordering: axis-0=v, axis-1=h
                n0 = max(3, 2 * int(smooth_extent / dy) + 1)  # v direction (axis-0)
                n1 = max(3, 2 * int(smooth_extent / dx) + 1)  # h direction (axis-1)
                kv = np.linspace(-smooth_extent, smooth_extent, n0)
                kh = np.linspace(-smooth_extent, smooth_extent, n1)
                # indexing='xy': KH.shape = (n0, n1), axis-0 → kv, axis-1 → kh
                KH, KV = np.meshgrid(kh, kv)
                K = np.asarray(smooth_kernel(KH, KV), dtype=float)
            else:
                K = np.asarray(smooth_kernel, dtype=float)
            total = K.sum()
            if total > 0:
                K /= total
            # convolve() does not support per-axis modes — reflect-pad the r⊥
            # boundary manually for cylindrical, then use constant elsewhere.
            if isinstance(smooth_mode, list):
                r_ax = smooth_mode.index('reflect')
                pad = K.shape[r_ax] // 2
                pads = [(0, 0)] * data.ndim
                pads[r_ax] = (pad, 0)   # only the r⊥=0 side needs reflection
                data_pad = np.pad(data, pads, mode='reflect')
                result   = convolve(data_pad, K, mode='constant', cval=0.0)
                slices   = [slice(None)] * data.ndim
                slices[r_ax] = slice(pad, None)
                data = result[tuple(slices)]
            else:
                data = convolve(data, K, mode=smooth_mode, cval=0.0)

    # Apply the mincount mask after normalisation/smoothing so under-sampled
    # bins are hidden (NaN → transparent for imshow/pcolormesh/contour*)
    # without having distorted the convolution.
    if below_mincount is not None:
        data = np.where(below_mincount, np.nan, data)

    # Colour scale — set after smoothing so vmin/vmax track the smoothed data.
    # nan-aware reductions so masked (mincount) bins don't turn the scale nan.
    if symmetric:
        abs_max = float(np.nanmax(np.abs(data)))
        vmax_ = abs_max if vmax is None else float(vmax)
        vmin_ = -vmax_  if vmin is None else float(vmin)
    else:
        vmax_ = float(np.nanmax(data)) if vmax is None else float(vmax)
        vmin_ = min(float(np.nanmin(data)), 0.0) if vmin is None else float(vmin)

    if vcenter is not None:
        norm = mcolors.TwoSlopeNorm(vcenter=float(vcenter), vmin=vmin_, vmax=vmax_)
    elif vmin_ < 0 < vmax_:
        norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin_, vmax=vmax_)
    else:
        norm = mcolors.Normalize(vmin=vmin_, vmax=vmax_)

    if np.ndim(levels) > 0:
        lvl_vals = np.asarray(levels, dtype=float)
    else:
        lvl_vals = np.linspace(vmin_, vmax_, levels + 2)[1:-1]
    mappable = None

    # lvl_vals sits strictly inside (vmin_, vmax_) -- the endpoints are
    # dropped so the outermost band isn't a degenerate zero-area contour.
    # Without `extend`, contourf/contour leave anything past the outermost
    # *drawn* level completely unfilled (background shows through) rather
    # than clamped to the colormap's end colour, which silently hides
    # exactly the peak bins a vmax cap is meant to highlight.
    #
    # Only extend on the high side: these maps are mostly bulk-density/
    # near-zero background below the lowest level (routinely >80% of
    # bins), and leaving that unfilled -- rather than flooding it with the
    # colormap's "under" colour via extend='min'/'both' -- is the existing,
    # deliberate look (only structure above the noise floor gets painted).
    finite = data[np.isfinite(data)]
    has_over = finite.size > 0 and np.nanmax(finite) > lvl_vals[-1]
    extend = 'max' if has_over else 'neither'

    if render == 'image':
        imshow_kw = dict(origin='lower',
                         extent=[h_edges[0], h_edges[-1], v_edges[0], v_edges[-1]],
                         aspect='auto', cmap=cmap, norm=norm)
        if interpolation is not None:
            imshow_kw['interpolation'] = interpolation
        mappable = ax.imshow(data, **imshow_kw)
    elif render == 'fill':
        mappable = ax.pcolormesh(
            h_edges, v_edges, data,
            cmap=cmap, norm=norm, shading='flat', rasterized=True,
        )
    elif render == 'contourf':
        mappable = ax.contourf(hc, vc, data, levels=lvl_vals,
                               cmap=cmap, norm=norm, extend=extend)
    elif render == 'contour':
        mappable = ax.contour(hc, vc, data, levels=lvl_vals,
                              cmap=cmap, norm=norm, linewidths=1.0, extend=extend)
    elif render == 'fill+contour':
        mappable = ax.pcolormesh(
            h_edges, v_edges, data,
            cmap=cmap, norm=norm, shading='flat', rasterized=True,
        )
        ax.contour(hc, vc, data, levels=lvl_vals,
                   colors='white', linewidths=0.7, alpha=0.7, extend=extend)
    elif render == 'contourf+contour':
        mappable = ax.contourf(hc, vc, data, levels=lvl_vals,
                               cmap=cmap, norm=norm, extend=extend)
        ax.contour(hc, vc, data, levels=lvl_vals,
                   colors='white', linewidths=0.7, alpha=0.7, extend=extend)
    else:
        raise ValueError(
            f"render must be one of: 'image', 'fill', 'contourf', 'contour', "
            f"'fill+contour', 'contourf+contour'. Got {render!r}."
        )

    if colorbar and mappable is not None:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb  = fig.colorbar(mappable, cax=cax)
        if colorbar_label is False:
            pass
        elif colorbar_label is not None:
            cb.set_label(colorbar_label)
        else:
            if normalize == 'density' and density_reference is not None:
                cb_label = r'$\rho\,/\,\rho_{\mathrm{bulk}}$'
            elif normalize == 'density':
                cb_label = ('counts Å$^{-3}$'
                            if proj in ('cylindrical', 'spherical')
                            else 'counts Å$^{-2}$')
            elif normalize == 'max':
                cb_label = 'normalised density'
            else:
                cb_label = 'counts'
            cb.set_label(cb_label)

    # Axis labels — planar uses Cartesian names; cylindrical/spherical use
    # physical names.  bond_vertical swaps which label goes on which axis.
    _label_map = {
        'xy':          ('x (Å)',  'y (Å)'),
        'xz':          ('x (Å)',  'z (Å)'),
        'yz':          ('y (Å)',  'z (Å)'),
        'cylindrical': ('z (Å)',  'r⊥ (Å)'),
        'spherical':   ('r (Å)',  'θ (rad)'),
    }
    xl, yl = _label_map.get(proj, ('(Å)', '(Å)'))
    if bond_vertical:
        xl, yl = yl, xl
    ax.set_xlabel(xlabel if xlabel is not None else xl)
    ax.set_ylabel(ylabel if ylabel is not None else yl)

    if proj in _PROJ_AXES or proj == 'cylindrical':
        ax.set_aspect('equal')

    # Flip-bond axis inversion — which axis carries the bond direction differs
    # by projection type.
    # Planar:      bond on y-axis (bv=False) or x-axis (bv=True)
    # Cylindrical: bond/z on x-axis (bv=False) or y-axis (bv=True)  ← inverted!
    # Spherical:   flip θ direction; θ on y-axis (bv=False) or x-axis (bv=True)
    if flip_bond:
        if proj == 'cylindrical':
            if bond_vertical: ax.invert_yaxis()
            else:             ax.invert_xaxis()
        else:
            # planar and spherical: same axis convention
            if bond_vertical: ax.invert_xaxis()
            else:             ax.invert_yaxis()

    if grid:
        gkw = dict(color='white', alpha=0.35, linewidth=0.7, linestyle='-')
        if grid_kw:
            gkw.update(grid_kw)
        if proj == 'cylindrical':
            # A rectilinear grid can't show constant-distance-from-centre on a
            # (z, r⊥) map, so draw concentric rings instead: every point on a
            # ring of radius R satisfies sqrt(z² + r⊥²) = R, i.e. a spherical
            # shell around the centre atom.  r⊥ ≥ 0 clips each ring to a
            # semicircle.
            from matplotlib.patches import Circle
            step = grid_step if grid_step is not None else 1.0
            gkw.setdefault('zorder', 2)
            r_max = float(max(np.abs(h_edges).max(), np.abs(v_edges).max()))
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            n_rings = int(np.floor(r_max / step))
            for i in range(1, n_rings + 1):
                ax.add_patch(Circle((0.0, 0.0), radius=i * step,
                                     fill=False, clip_on=True, **gkw))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        else:
            gkw.setdefault('zorder', 2)
            if grid_step is not None:
                from matplotlib.ticker import MultipleLocator
                ax.xaxis.set_major_locator(MultipleLocator(grid_step))
                ax.yaxis.set_major_locator(MultipleLocator(grid_step))
            ax.set_axisbelow(False)
            ax.grid(True, **gkw)

    if molecule is not None:
        # Freeze limits so atom/bond artists don't expand the axes range.
        # Passed into _draw_molecule_overlay (rather than it re-querying
        # ax.get_xlim()/get_ylim() itself) because the bond-line ax.plot()
        # calls inside it re-trigger matplotlib's autoscale with the
        # standard ~5% margin, transiently *widening* the axes -- a
        # highlight positioned to fit that transient range then ends up
        # outside the range we restore to right after, and gets clipped
        # away entirely (this was a real bug: the specular highlight on
        # an atom sitting near the axes edge, e.g. the centre atom at
        # r_perp=0 on a cylindrical map, silently vanished with
        # render='contourf' specifically, because contourf's own default
        # margin differs from plot()'s and so happened to disagree with
        # it more often than pcolormesh's did).
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        # axis inversion (done above) already repositions the molecule in data
        # coordinates — passing flip_bond here would double-flip it
        _draw_molecule_overlay(ax, molecule, mol_scale, bond_vertical, False,
                               xlim=xlim, ylim=ylim)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    return fig, ax


_BOND_AXIS = 2   # bond is always along z after _bond_rotation


def _draw_molecule_overlay(ax, molecule, mol_scale, bond_vertical=False, flip_bond=False,
                           xlim=None, ylim=None):
    """Draw the molecule directly on *ax* in data coordinates.

    Atoms are drawn as circles via draw_atom_2d; bonds as grey lines between
    the centre atom and each other atom.  Using the main axes directly (rather
    than an inset) means:
      - atoms are clipped at the axes boundary automatically (no tick overlap)
      - secondary atoms (e.g. the second H of H₂O) appear at the correct
        projected position regardless of projection type
      - flip_bond works by negating the bond-axis coordinate before projection

    Parameters
    ----------
    xlim, ylim : tuple of float, optional
        The axes limits the caller will actually keep (frozen *before*
        this function runs, since the bond-line ax.plot() calls below
        re-trigger matplotlib's autoscale and transiently change the
        limits). Used to decide which side to place each atom's specular
        highlight on -- reading ax.get_xlim()/get_ylim() *inside* this
        function instead would see that transient, about-to-be-discarded
        range rather than the one actually in effect in the final figure,
        and can place the highlight somewhere that gets clipped away
        entirely. Falls back to a live query if not given.
    """
    from .colours import draw_atom_2d

    proj = molecule.get('projection', 'xz')
    ax0, ax1 = _PROJ_AXES.get(proj, (0, 2))
    sign = -1.0 if flip_bond else 1.0

    def _proj(p):
        p = np.asarray(p, dtype=float).copy()
        p[_BOND_AXIS] *= sign
        if proj == 'cylindrical':
            z = float(p[2])
            r = float(np.sqrt(p[0] ** 2 + p[1] ** 2))
            return (r, z) if bond_vertical else (z, r)
        if proj == 'spherical':
            r = float(np.linalg.norm(p))
            if r < 1e-10:
                return (0.0, 0.0)
            theta = float(np.arccos(np.clip(p[2] / r, -1.0, 1.0)))
            return (theta, r) if bond_vertical else (r, theta)
        # planar
        if bond_vertical:
            return (float(p[ax1]), float(p[ax0]))
        return (float(p[ax0]), float(p[ax1]))

    positions_2d = [_proj(p) for p in molecule['positions_3d']]
    symbols      = molecule['symbols']

    # Bond lines from centre atom to every other atom (drawn first, under circles)
    if len(positions_2d) > 1:
        x0, y0 = positions_2d[0]
        for (x1, y1) in positions_2d[1:]:
            ax.plot([x0, x1], [y0, y1], '-', color='#444444',
                    lw=1.5, solid_capstyle='round', zorder=4, clip_on=True)

    # Atom circles — drawn on ax directly so clip_on=True clips at axes boundary.
    # An atom sitting right against an axes edge (e.g. the centre atom at
    # r⊥=0/r=0 on cylindrical/spherical maps) would have its default
    # upper-left specular highlight clipped off; flip it to whichever side
    # actually stays inside the visible limits.
    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()

    def _fits(pos, offset, lo, hi):
        return lo <= pos + offset <= hi

    for (x, y), sym in zip(positions_2d, symbols):
        r = _elem_radius(sym) * mol_scale
        hdx = -0.3 if _fits(x, -0.3 * r, xlim[0], xlim[1]) else 0.3
        hdy = 0.3 if _fits(y, 0.3 * r, ylim[0], ylim[1]) else -0.3
        draw_atom_2d(ax, x, y, sym, scale=mol_scale, zorder_base=5,
                     highlight_dir=(hdx, hdy))


def _unpack_sdf_result(value):
    """Accept either a raw hists dict or the (hists, meta) tuple."""
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], dict):
        return value  # (hists, meta)
    return value, {}  # hists only — synthesise empty meta


def plot_sdf_maps(
    data,
    center_symbol=None,
    align_symbol=None,
    secondary_symbol=None,
    angle_deg=None,
    bisector=False,
    mol_scale=1.0,
    projections=None,
    figsize_per_panel=(5, 5),
    **plot_kw,
):
    """Plot all projections from :func:`spatial_density_maps` in a single row.

    Parameters
    ----------
    data : tuple or dict
        Raw ``(hists, meta)`` tuple from :func:`spatial_density_maps`, or
        just the ``hists`` dict.
    center_symbol : str or None
        Element symbol of the centre atom (e.g. ``'O'``).  Together with
        *align_symbol*, triggers automatic molecule overlay using
        ``meta['mean_bond_len']``.
    align_symbol : str or None
        Element symbol of the bond atom (e.g. ``'H'``).
    mol_scale : float
        Multiplier applied to covalent radii in the overlay.
    figsize_per_panel : tuple
        Width × height in inches per panel.
    **plot_kw
        Forwarded to :func:`plot_sdf` for every panel.

    Returns
    -------
    fig : Figure
    axs : dict
        ``{projection: ax}``
    """
    import matplotlib.pyplot as plt

    hists, meta = _unpack_sdf_result(data)
    bond_len    = meta.get('mean_bond_len')
    projections = ([projections] if isinstance(projections, str) else
                   list(projections) if projections is not None else list(hists))
    n = len(projections)
    w, h = figsize_per_panel
    fig, axes = plt.subplots(1, n, figsize=(w * n, h), squeeze=False)
    axs = {}
    for ax, proj in zip(axes[0], projections):
        hist, xe, ye = hists[proj]
        mol = None
        if center_symbol and align_symbol and bond_len:
            mol = molecule_overlay(center_symbol, align_symbol, bond_len,
                                   projection=proj,
                                   secondary_symbol=secondary_symbol,
                                   angle_deg=angle_deg,
                                   bisector=bisector)
        plot_sdf(hist, xe, ye, ax=ax, projection=proj,
                 molecule=mol, mol_scale=mol_scale, **plot_kw)
        ax.set_title(proj)
        axs[proj] = ax
    fig.tight_layout()
    return fig, axs


def plot_sdf_panel(
    panels,
    center_symbol=None,
    align_symbol=None,
    secondary_symbol=None,
    angle_deg=None,
    bisector=False,
    mol_scale=1.0,
    projections=None,
    row_labels=None,
    figsize_per_panel=(5, 5),
    **plot_kw,
):
    """Grid of SDF panels: rows = targets, columns = projections.

    Parameters
    ----------
    panels : dict
        ``{label: result}`` where each *result* is the raw ``(hists, meta)``
        tuple from :func:`spatial_density_maps` or just the ``hists`` dict.
    center_symbol : str or None
        Element symbol of the centre atom.  Together with *align_symbol*,
        triggers automatic molecule overlay using each panel's
        ``meta['mean_bond_len']``.
    align_symbol : str or None
        Element symbol of the bond atom.
    mol_scale : float
        Multiplier applied to covalent radii in the overlay.
    row_labels : list of str or None
        Explicit row titles; defaults to the panel dict keys.
    figsize_per_panel : tuple
        Width × height in inches per panel.
    **plot_kw
        Forwarded to every :func:`plot_sdf` call.

    Returns
    -------
    fig : Figure
    axs : ndarray, shape (n_labels, n_projections)
    """
    import matplotlib.pyplot as plt

    labels         = list(panels)
    first_hists, _ = _unpack_sdf_result(next(iter(panels.values())))
    projections    = ([projections] if isinstance(projections, str) else
                      list(projections) if projections is not None
                      else list(first_hists))
    n_row = len(labels)
    n_col = len(projections)
    w, h  = figsize_per_panel

    fig, axes = plt.subplots(n_row, n_col,
                              figsize=(w * n_col, h * n_row),
                              squeeze=False)

    for r, label in enumerate(labels):
        hists, meta = _unpack_sdf_result(panels[label])
        bond_len    = meta.get('mean_bond_len')
        for c, proj in enumerate(projections):
            ax           = axes[r, c]
            hist, xe, ye = hists[proj]
            mol = None
            if center_symbol and align_symbol and bond_len:
                mol = molecule_overlay(center_symbol, align_symbol, bond_len,
                                       projection=proj,
                                       secondary_symbol=secondary_symbol,
                                       angle_deg=angle_deg,
                                       bisector=bisector)
            plot_sdf(hist, xe, ye, ax=ax, projection=proj,
                     molecule=mol, mol_scale=mol_scale, **plot_kw)
            if r == 0:
                ax.set_title(proj)
            if c == 0:
                row_title = row_labels[r] if row_labels else label
                ax.set_ylabel(row_title + '\n' + ax.get_ylabel())

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 3D accumulation and isosurface plotting
# ---------------------------------------------------------------------------

def spatial_density_3d(
    frames,
    center,
    align_to,
    target,
    secondary=None,
    half_space=None,
    half_space_offset=0.0,
    rmax=5.0,
    bins=50,
    stride=1,
    weights=None,
    exclude_align=True,
):
    """Accumulate a 3D spatial density volume in the bond-aligned frame.

    The bond vector is rotated onto +z; the returned volume is indexed as
    ``vol[ix, iy, iz]`` where iz is along the bond.  All three axes span
    ``[-rmax, +rmax]``.

    Parameters
    ----------
    frames : list of Frame
    center : str or dict
        Selector for the anchor atom.
    align_to : str or dict
        Selector for the atom whose direction defines +z.
    target : str or dict
        Selector for atoms to accumulate.
    secondary : str or dict or None
        Selector for a second atom in the *center* molecule used to fix the
        rotation around z (places it in the xz half-plane).  Required to
        recover azimuthal structure for non-linear molecules.
    half_space : str or None
        Restrict to one half of space before binning.  Same spec as in
        :func:`spatial_density_map`.
    half_space_offset : float
        Shift the half-space cut plane (Å).
    rmax : float
        Half-extent of each axis in Å.
    bins : int
        Number of bins per axis.  The volume uses ``bins³`` memory; 50 is a
        practical default.  Increase to 80–100 for smoother surfaces with
        long trajectories.
    stride : int
        Process every *n*-th frame.
    weights : str or None
        Per-atom scalar weight key in ``frame.atoms.arrays`` (e.g.
        ``'spins'``, ``'charges'``).  Each bin accumulates the sum of
        weights rather than a count.  Attach the companion HDF5 first via
        ``traj.build(companion=...)``.

    Returns
    -------
    vol : ndarray, shape (bins, bins, bins)
        Raw accumulated volume (counts or weight sums).
    xedges, yedges, zedges : ndarray, shape (bins+1,)
        Bin edges along x, y, z.
    meta : dict
        ``'n_center'``, ``'mean_bond_len'``, ``'half_space'``.
    """
    edges = np.linspace(-rmax, rmax, bins + 1)
    xedges = yedges = zedges = edges
    vol = np.zeros((bins, bins, bins), dtype=np.float64)
    bond_lens = []
    n_center = 0

    for frame in frames[::stride]:
        c_idx = resolve(frame, center)
        a_idx = resolve(frame, align_to)
        t_idx = resolve(frame, target)

        if len(c_idx) == 0 or len(a_idx) == 0 or len(t_idx) == 0:
            continue

        pos  = frame.positions
        cell = np.asarray(frame.atoms.get_cell())
        a_set = set(a_idx.tolist())

        for ci in c_idx:
            mol_id   = int(frame.atom_to_mol[ci])
            mol_atoms = frame.molecules[mol_id]
            a_in_mol = [a for a in mol_atoms if a in a_set]
            if not a_in_mol:
                continue
            ai = a_in_mol[0]

            bd, bl = find_mic((pos[ai] - pos[ci])[None], cell, pbc=True)
            bond_lens.append(float(bl[0]))
            R_mat = _bond_rotation(bd[0])

            # Secondary alignment: fix rotation around z
            if secondary is not None:
                s_idx = resolve(frame, secondary)
                s_set = set(s_idx.tolist())
                s_in_mol = [a for a in mol_atoms if a in s_set and a != ai]
                if s_in_mol:
                    sd, _ = find_mic((pos[s_in_mol[0]] - pos[ci])[None],
                                     cell, pbc=True)
                    sec = R_mat @ sd[0]
                    phi = np.arctan2(sec[1], sec[0])
                    c_, s_ = np.cos(-phi), np.sin(-phi)
                    R_z = np.array([[c_, -s_, 0.], [s_, c_, 0.], [0., 0., 1.]])
                    R_mat = R_z @ R_mat

            diff   = pos[t_idx] - pos[ci]
            mic, _ = find_mic(diff, cell, pbc=True)
            excl   = (t_idx == ci) | (t_idx == ai)
            active = ~excl

            w = None
            if weights is not None:
                w = _get_weights(frame, weights, t_idx, active)

            rot = (R_mat @ mic[active].T).T

            if half_space is not None:
                hs_ax, hs_pos = _HS_SPEC[half_space]
                cut  = float(half_space_offset)
                mask = (rot[:, hs_ax] > cut) if hs_pos else (rot[:, hs_ax] < -cut)
                rot  = rot[mask]
                if w is not None:
                    w = w[mask]

            h, _ = np.histogramdd(rot, bins=[xedges, yedges, zedges], weights=w)
            vol += h
            n_center += 1

    return vol, xedges, yedges, zedges, {
        'n_center':      n_center,
        'mean_bond_len': float(np.mean(bond_lens)) if bond_lens else 0.,
        'half_space':    half_space,
    }


def plot_sdf_3d(
    vol,
    xedges=None,
    yedges=None,
    zedges=None,
    ax=None,
    level=None,
    sigma=None,
    normalize='density',
    n_center=None,
    cmap='inferno',
    alpha=0.5,
    colorbar=True,
    **kwargs,
):
    """Plot one or more isosurfaces of a 3D spatial density volume.

    Accepts either the raw return value of :func:`spatial_density_3d` as the
    first argument (5-tuple ``(vol, xe, ye, ze, meta)``) or the arrays
    separately.

    Parameters
    ----------
    vol : ndarray, shape (N, N, N) or 5-tuple
        3D density volume from :func:`spatial_density_3d`, or the full
        ``(vol, xedges, yedges, zedges, meta)`` tuple.
    xedges, yedges, zedges : ndarray
        Bin edges — inferred from tuple if *vol* is a tuple.
    ax : matplotlib Axes3D or None
        Target axes.  A new figure with a 3D subplot is created when None.
    level : float, list of float, or None
        Isosurface level(s) in the normalised density units.  Multiple values
        produce nested surfaces coloured along *cmap*.  ``None`` auto-selects
        at the 80th percentile of non-zero voxels after smoothing.
    sigma : float or None
        Gaussian smoothing radius in Å applied before contouring.  Does not
        modify *vol* in-place.  Typical range 0.2–0.8 Å.
    normalize : str or None
        ``'density'`` — divide by voxel volume and *n_center* to get Å⁻³.
        ``None`` — plot raw counts / weight sums.
    n_center : int or None
        Number of centre atoms used; inferred from *meta* when *vol* is a
        tuple.
    cmap : str
        Colormap used to colour multiple isosurface levels.
    alpha : float
        Surface opacity (0 transparent, 1 opaque).
    colorbar : bool
        Add a colorbar mapping level value to colour.
    **kwargs
        Forwarded to :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection`.

    Returns
    -------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        raise ImportError(
            "scikit-image is required for isosurface plotting: "
            "pip install scikit-image"
        )
    from scipy.ndimage import gaussian_filter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    cmap = _resolve_cmap(cmap)  # resolve + validate; substitutes tol.X aliases

    # Unpack tuple input
    _meta = {}
    if isinstance(vol, tuple) and len(vol) == 5 and xedges is None:
        vol, xedges, yedges, zedges, _meta = vol
    elif isinstance(vol, tuple) and len(vol) == 4 and xedges is None:
        vol, xedges, yedges, zedges = vol

    if n_center is None:
        n_center = _meta.get('n_center', 1)

    data = vol.astype(float).copy()

    # Normalize
    if normalize == 'density' and data.sum() > 0:
        dx = float(xedges[1] - xedges[0])
        dy = float(yedges[1] - yedges[0])
        dz = float(zedges[1] - zedges[0])
        data /= float(max(n_center, 1)) * dx * dy * dz

    # Gaussian smoothing
    if sigma is not None and sigma > 0:
        dx = float(xedges[1] - xedges[0])
        data = gaussian_filter(data, sigma=sigma / dx)

    # Auto-level
    if level is None:
        nonzero = data[data > 0]
        level = float(np.percentile(nonzero, 80)) if nonzero.size else 1.0
    levels = np.atleast_1d(level).ravel().tolist()

    # Axes
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Bin centres (marching_cubes spacing argument sets voxel size)
    dx = float(xedges[1] - xedges[0])
    dy = float(yedges[1] - yedges[0])
    dz = float(zedges[1] - zedges[0])
    x0, y0, z0 = float(xedges[0]) + dx/2, float(yedges[0]) + dy/2, float(zedges[0]) + dz/2

    cmap_obj   = cm.get_cmap(cmap)
    lmin, lmax = min(levels), max(levels)
    norm       = mcolors.Normalize(vmin=lmin, vmax=lmax)

    for lv in sorted(levels):
        try:
            verts, faces, _, _ = marching_cubes(data, level=lv,
                                                spacing=(dx, dy, dz))
        except ValueError:
            continue
        # Shift vertices from voxel-index space to Å
        verts[:, 0] += x0
        verts[:, 1] += y0
        verts[:, 2] += z0

        color = cmap_obj(norm(lv))
        mesh  = Poly3DCollection(verts[faces], alpha=alpha, **kwargs)
        mesh.set_facecolor(color)
        mesh.set_edgecolor('none')
        ax.add_collection3d(mesh)

    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(yedges[0], yedges[-1])
    ax.set_zlim(zedges[0], zedges[-1])
    ax.set_xlabel('x (Å)')
    ax.set_ylabel('y (Å)')
    ax.set_zlabel('z (Å)')

    if colorbar and len(levels) > 1:
        sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        ax.get_figure().colorbar(sm, ax=ax, shrink=0.6, label='density (Å⁻³)')

    return ax


# ---------------------------------------------------------------------------
# Axes extension
# ---------------------------------------------------------------------------
import matplotlib.axes as _mpl_axes


def _ax_plot_sdf(self, hist, xedges=None, yedges=None, **kwargs):
    return plot_sdf(hist, xedges, yedges, ax=self, **kwargs)


_mpl_axes.Axes.plot_sdf = _ax_plot_sdf
