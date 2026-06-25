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
        'folds':       {},
        'description': 'linear, polar — e.g. OH, CO, HCN',
    },
    'Dinfh': {
        'projections': ('cylindrical', 'spherical'),
        'secondary':   False,
        'folds':       {'z': 'abs'},          # +z ≡ −z
        'description': 'linear, centrosymmetric — e.g. CO₂, N₂, O₂',
    },
    'C2v': {
        'projections': ('xz', 'yz', 'cylindrical'),
        'secondary':   True,
        'folds':       {'x': 'abs', 'y': 'abs'},   # σv(yz) and σv(xz)
        'description': 'bent / pyramidal — e.g. H₂O, SO₂',
    },
    'C3v': {
        'projections': ('xz', 'cylindrical'),
        'secondary':   True,
        'folds':       {'x': 'abs'},               # σv mirrors
        'description': 'pyramidal, 3-fold — e.g. NH₃, CHCl₃',
    },
    'Cs': {
        'projections': ('xz', 'yz', 'xy'),
        'secondary':   True,
        'folds':       {'y': 'abs'},               # single σ plane
        'description': 'single mirror plane',
    },
    'C1': {
        'projections': ('xz', 'yz', 'xy'),
        'secondary':   True,
        'folds':       {},
        'description': 'no symmetry',
    },
    'Td': {
        'projections': ('spherical',),
        'secondary':   True,
        'folds':       {},
        'description': 'tetrahedral — e.g. CH₄, CCl₄',
    },
    'Oh': {
        'projections': ('spherical',),
        'secondary':   True,
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
            excl = (t_idx == ci) | (t_idx == ai)
            active = ~excl
            mic = mic[active]

            # Per-atom weights for active targets
            w = None
            if weights is not None:
                w = np.asarray(frame.atoms.arrays[weights])[t_idx[active]]

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

    return hist, xedges, yedges, {
        'n_center':      n_center,
        'mean_bond_len': float(np.mean(bond_lens)) if bond_lens else 0.,
        'projection':    projection,
        'half_space':    half_space,
    }


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
    """
    if symmetry not in SCHOENFLIES:
        raise ValueError(
            f"symmetry must be one of {sorted(SCHOENFLIES)}. Got {symmetry!r}."
        )

    sym         = SCHOENFLIES[symmetry]
    projections = sym['projections']
    folds       = sym['folds']

    if sym['secondary'] and secondary is None:
        raise ValueError(
            f"symmetry={symmetry!r} requires a secondary selector to fix the "
            "rotation around the bond axis (z)."
        )

    edges = {p: _sym_edges(p, folds, rmax, bins) for p in projections}
    hists = {p: np.zeros((bins, bins), dtype=np.float64) for p in projections}
    bond_lens = []
    n_center  = 0

    for frame in frames[::stride]:
        c_idx = resolve(frame, center)
        a_idx = resolve(frame, align_to)
        t_idx = resolve(frame, target)
        s_idx = resolve(frame, secondary) if secondary is not None else None

        if len(c_idx) == 0 or len(a_idx) == 0 or len(t_idx) == 0:
            continue

        pos   = frame.positions
        cell  = np.asarray(frame.atoms.get_cell())
        a_set = set(a_idx.tolist())
        s_set = set(s_idx.tolist()) if s_idx is not None else set()

        for ci in c_idx:
            mol_id    = int(frame.atom_to_mol[ci])
            mol_atoms = frame.molecules[mol_id]

            # Primary alignment: bond → +z
            a_in_mol = [a for a in mol_atoms if a in a_set]
            if not a_in_mol:
                continue
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
            excl   = (t_idx == ci) | (t_idx == ai)
            active = ~excl
            rot    = (R_mat @ mic[active].T).T

            # Per-atom weights for active targets
            w = None
            if weights is not None:
                w = np.asarray(frame.atoms.arrays[weights])[t_idx[active]]

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
    }
    return {p: (hists[p], *edges[p]) for p in projections}, meta


def molecule_overlay(center_symbol, align_symbol, bond_len, projection='xz'):
    """Build a molecule dict for use as the *molecule* argument of :func:`plot_sdf`.

    Places *center_symbol* at the origin and *align_symbol* at
    ``(bond_len, 0, 0)`` in the aligned frame.

    Parameters
    ----------
    center_symbol : str
        Element symbol of the centre atom (e.g. ``'O'``).
    align_symbol : str
        Element symbol of the bond atom (e.g. ``'H'``).
    bond_len : float
        Bond length in Å (use ``meta['mean_bond_len']`` from
        :func:`spatial_density_map`).
    projection : str
        Must match the projection used when computing the histogram.

    Returns
    -------
    dict
    """
    return {
        'symbols':      [center_symbol, align_symbol],
        'positions_3d': [np.zeros(3), np.array([0., 0., bond_len])],
        'projection':   projection,
    }


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
    normalize='density',
    molecule=None,
    mol_scale=1.0,
    bond_vertical=False,
    flip_bond=False,
    colorbar=True,
    xlabel=None,
    ylabel=None,
    projection=None,
    center_symbol=None,
    align_symbol=None,
    n_center=None,
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
    levels : int
        Number of contour levels.
    vmax : float or None
        Colour-scale ceiling.  Defaults to the histogram maximum after
        normalisation.
    normalize : str or None
        ``'density'`` — divide by bin area → counts / Å².
        ``'max'``     — scale peak to 1.
        ``None``      — raw counts.
    molecule : dict or None
        Central molecule overlay produced by :func:`molecule_overlay`.
    mol_scale : float
        Multiplier applied to covalent radii for the overlay circles.
    bond_vertical : bool
        ``False`` (default) — bond axis runs horizontally (left→right).
        ``True``            — bond axis runs vertically (bottom→top),
        perpendicular axis is horizontal.
    flip_bond : bool
        Reverse the bond axis direction.  With ``bond_vertical=False`` this
        inverts the x-axis (bond runs right→left); with ``bond_vertical=True``
        it inverts the y-axis (bond runs top→bottom), placing the align-to
        atom below the centre atom.
    colorbar : bool
        Add a colorbar.
    xlabel, ylabel : str or None
        Axis labels — auto-generated from projection if None.

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

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
            molecule = molecule_overlay(center_symbol, align_symbol, bond_len,
                                        projection=projection or 'xz')

    # n_center: explicit argument wins, then meta, then 1 (raw counts).
    if n_center is None:
        n_center = _meta.get('n_center', 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    # Derive projection early — needed for normalization and labels.
    # Explicit projection= wins; fall back to the molecule dict, then 'xz'.
    if projection is not None:
        proj = projection
    elif molecule is not None:
        proj = molecule.get('projection', 'xz')
    else:
        proj = 'xz'

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
    elif normalize == 'max' and data.max() > 0:
        data = data / data.max()

    vmax_ = float(data.max()) if vmax is None else vmax
    norm  = mcolors.Normalize(vmin=0, vmax=vmax_)
    lvl_vals = np.linspace(0., vmax_, levels + 2)[1:-1]
    mappable = None

    if render == 'fill':
        mappable = ax.pcolormesh(
            h_edges, v_edges, data,
            cmap=cmap, norm=norm, shading='flat', rasterized=True,
        )
    elif render == 'contourf':
        mappable = ax.contourf(hc, vc, data, levels=lvl_vals,
                               cmap=cmap, norm=norm)
    elif render == 'contour':
        mappable = ax.contour(hc, vc, data, levels=lvl_vals,
                              cmap=cmap, norm=norm, linewidths=1.0)
    elif render == 'fill+contour':
        mappable = ax.pcolormesh(
            h_edges, v_edges, data,
            cmap=cmap, norm=norm, shading='flat', rasterized=True,
        )
        ax.contour(hc, vc, data, levels=lvl_vals,
                   colors='white', linewidths=0.7, alpha=0.7)
    elif render == 'contourf+contour':
        mappable = ax.contourf(hc, vc, data, levels=lvl_vals,
                               cmap=cmap, norm=norm)
        ax.contour(hc, vc, data, levels=lvl_vals,
                   colors='white', linewidths=0.7, alpha=0.7)
    else:
        raise ValueError(
            f"render must be one of: 'fill', 'contourf', 'contour', "
            f"'fill+contour', 'contourf+contour'. Got {render!r}."
        )

    if colorbar and mappable is not None:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb  = fig.colorbar(mappable, cax=cax)
        if normalize == 'density':
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

    if molecule is not None:
        _draw_molecule_overlay(ax, molecule, mol_scale, bond_vertical, flip_bond)

    return fig, ax


_BOND_AXIS = 2   # bond is always along z after _bond_rotation


def _draw_molecule_overlay(ax, molecule, mol_scale, bond_vertical=False, flip_bond=False):
    """Draw the molecule in a transparent inset axes centred at the data origin.

    Using an inset axes (positioned in data coordinates) decouples the molecule
    rendering from the main axes direction, making flip_bond reliable regardless
    of how pcolormesh/contourf edges are oriented.
    """
    from .colours import draw_atom_2d, elem_radius as _er

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

    pts    = np.array(positions_2d)
    radii  = np.array([_er(s) * mol_scale for s in symbols])
    r_max  = float(radii.max())
    margin = 1.3

    if proj in ('cylindrical', 'spherical'):
        # Atoms lie exactly on the domain boundary (r⊥=0 or θ=0).
        # Anchor the inset AT that boundary and extend into the plot only —
        # never centre across it, or half the inset falls outside the axes.
        cx       = float(pts[:, 0].mean())
        cy       = float(pts[:, 1].mean())
        lim_par  = (float(np.abs(pts[:, 0] - cx).max()) + r_max) * margin
        lim_perp = r_max * margin
        if bond_vertical:
            # z on y-axis, r⊥ on x-axis; atoms sit on left boundary (x=0)
            x0, y0 = 0,            cy - lim_par
            w,  h  = 2 * lim_perp, 2 * lim_par
            inset  = ax.inset_axes([x0, y0, w, h], transform=ax.transData)
            inset.set_xlim(0, 2 * lim_perp)
            inset.set_ylim(cy - lim_par, cy + lim_par)
        else:
            # z on x-axis, r⊥ on y-axis; atoms sit on bottom boundary (y=0)
            x0, y0 = cx - lim_par, 0
            w,  h  = 2 * lim_par,  2 * lim_perp
            inset  = ax.inset_axes([x0, y0, w, h], transform=ax.transData)
            inset.set_xlim(cx - lim_par, cx + lim_par)
            inset.set_ylim(0, 2 * lim_perp)
    else:
        # Planar: centre inset at the data origin
        lim = (float(np.abs(pts).max()) + r_max) * margin
        inset = ax.inset_axes([-lim, -lim, 2 * lim, 2 * lim],
                               transform=ax.transData)
        inset.set_xlim(-lim, lim)
        inset.set_ylim(-lim, lim)

    inset.set_aspect('equal')
    inset.axis('off')
    inset.patch.set_alpha(0.0)
    for patch in inset.patches:
        patch.set_clip_on(False)

    for (x, y), sym in zip(positions_2d, symbols):
        draw_atom_2d(inset, x, y, sym, scale=mol_scale)
    for patch in inset.patches:
        patch.set_clip_on(False)


def _unpack_sdf_result(value):
    """Accept either a raw hists dict or the (hists, meta) tuple."""
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], dict):
        return value  # (hists, meta)
    return value, {}  # hists only — synthesise empty meta


def plot_sdf_maps(
    data,
    center_symbol=None,
    align_symbol=None,
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
                                   projection=proj)
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
                                       projection=proj)
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
# Axes extension
# ---------------------------------------------------------------------------
import matplotlib.axes as _mpl_axes


def _ax_plot_sdf(self, hist, xedges=None, yedges=None, **kwargs):
    return plot_sdf(hist, xedges, yedges, ax=self, **kwargs)


_mpl_axes.Axes.plot_sdf = _ax_plot_sdf
