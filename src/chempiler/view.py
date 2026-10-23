"""Inline 3D structure viewer for Jupyter notebooks."""

from ase import Atoms


def show(structures, labels=None, index=0):
    """Display structures interactively using ASE's x3d viewer inline in Jupyter.

    Parameters
    ----------
    structures : Atoms | str | list | dict
        Anything structure-like:

        - ``Atoms`` — a single structure.
        - ``str`` — path to an XYZ / extXYZ file (frame 0 is read).
        - ``list`` of Atoms or paths — displayed in order.
        - ``dict`` — the ``envs`` dict returned by
          ``ChempilerTrajectory.rdf_peak_environments``:
          keys are peak distances (float, used as labels), values are
          lists of Atoms (the first cluster is shown).

    labels : list of str, optional
        Override labels for list input.  Ignored for dict input (labels
        come from the keys).
    """
    from IPython.display import display, HTML
    from ase.io import read as ase_read
    from ase.visualize import view as ase_view

    def _load(s):
        return ase_read(s) if isinstance(s, str) else s

    def _render(label, atoms):
        if label:
            display(HTML(f"<h4>{label}</h4>"))
        display(ase_view(atoms, viewer='x3d'))

    if isinstance(structures, dict):
        for key, val in structures.items():
            label = f"r = {key:.3f} Å" if isinstance(key, float) else str(key)
            atoms = _load(val[index] if isinstance(val, list) else val)
            _render(label, atoms)

    elif isinstance(structures, (list, tuple)):
        for i, s in enumerate(structures):
            label = labels[i] if labels and i < len(labels) else None
            _render(label, _load(s))

    elif isinstance(structures, str):
        label = labels[0] if labels else structures
        _render(label, _load(structures))

    elif isinstance(structures, Atoms):
        label = labels[0] if labels else None
        _render(label, structures)

    else:
        raise TypeError(f"Cannot display {type(structures).__name__}")


def _oriented_positions(positions, elev=0.0, azim=0.0, roll=0.0, center=None):
    """Rotate *positions* by (elev, azim, roll) -- a full 3-DOF orientation.

    Shared by :func:`view3d` and :func:`plot_structure` so that identical
    ``elev``/``azim``/``roll`` values give an identical view in both. The
    rotation is applied to the *data* once; each backend then renders from
    its own neutral default camera. This sidesteps trying to keep two
    unrelated camera systems in sync (3Dmol.js's internal quaternion
    camera -- which has no supported way to read its live state back into
    Python -- and matplotlib's own camera) by never needing to translate
    between them at all.

    ``roll`` matters more than it sounds like it should: free mouse
    rotation in :func:`view3d` has 3 rotational degrees of freedom, but
    ``elev``/``azim`` alone are only 2 -- any view reached by dragging that
    isn't a pure elevation+azimuth change (which is most of them) has a
    real roll component, and dropping it produces a visibly wrong
    reproduction, not just a rounding error.

    Parameters
    ----------
    positions : array-like, shape (N, 3)
    elev : float
        Degrees to tilt the view up/down, applied about the x-axis,
        second.
    azim : float
        Degrees to spin the view around the vertical (z) axis, applied
        first.
    roll : float
        Degrees to spin around the viewing axis itself (the new z-axis
        after the elev/azim rotations), applied third/last.
    center : array-like of shape (3,), optional
        Pivot point for the rotation. Defaults to the centroid.

    Returns
    -------
    numpy.ndarray, shape (N, 3)
    """
    import numpy as np

    positions = np.asarray(positions, dtype=float)
    c = positions.mean(axis=0) if center is None else np.asarray(center, dtype=float)
    p = positions - c

    az = np.radians(azim)
    el = np.radians(elev)
    ro = np.radians(roll)
    r_z_az = np.array([
        [np.cos(az), -np.sin(az), 0.],
        [np.sin(az),  np.cos(az), 0.],
        [0., 0., 1.],
    ])
    r_x_el = np.array([
        [1., 0., 0.],
        [0., np.cos(el), -np.sin(el)],
        [0., np.sin(el),  np.cos(el)],
    ])
    r_z_ro = np.array([
        [np.cos(ro), -np.sin(ro), 0.],
        [np.sin(ro),  np.cos(ro), 0.],
        [0., 0., 1.],
    ])
    r = r_z_ro @ r_x_el @ r_z_az
    return (r @ p.T).T + c


def _resolve_atom_ref(ref, mark_atoms):
    """Resolve one mark_bonds endpoint to an atom index.

    *ref* is either already an int index, or a custom label string
    matching one of *mark_atoms*'s values (when *mark_atoms* is a dict) --
    so ``mark_bonds`` can reference the same tags you gave atoms via
    ``mark_atoms={0: 'O3', 3: 'O1', ...}`` directly, e.g.
    ``mark_bonds=[('O3', 'O1')]``, instead of having to look their indices
    back up by hand.
    """
    if isinstance(ref, str) and isinstance(mark_atoms, dict):
        for idx, label in mark_atoms.items():
            if str(label) == ref:
                return idx
        raise KeyError(
            f"mark_bonds label {ref!r} not found among mark_atoms's labels "
            f"{list(mark_atoms.values())!r}"
        )
    return ref


def view3d(atoms, elev=0.0, azim=0.0, roll=0.0, mark_atoms=None, mark_bonds=None,
           bonds=False, bond_radius=0.08, sphere_scale=0.3, radius='covalent',
           show_orientation=False, width=500, height=400):
    """Interactive, rotatable 3D structure viewer built on py3Dmol (3Dmol.js).

    Unlike :func:`show` (ASE's x3d viewer, no annotation support), this
    renders *atoms* coloured with chempiler's own :data:`ELEM_COLOR`
    scheme rather than 3Dmol's defaults, and supports labelling individual
    atoms and drawing extra highlighted "bonds" between arbitrary atom
    pairs -- e.g. marking oxygen indices, or a non-covalent contact (a
    hydrogen bond, or the short O...O hemi-bond distance sometimes seen
    for the aqueous OH radical) that wouldn't be drawn as an ordinary bond.

    Ordinary covalent bonds are detected by 3Dmol.js itself from the raw
    coordinates (its standard distance-based heuristic for ball-and-stick
    rendering) rather than by chempiler's own bond perception -- this
    keeps the function simple, and 3Dmol's heuristic is perfectly
    adequate for the compact molecular clusters this is meant for (a
    handful to a few dozen atoms). Use *mark_bonds* for any contact you
    want drawn that this heuristic wouldn't pick up on its own.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to display.
    elev, azim, roll : float
        Starting orientation in degrees (see :func:`_oriented_positions`).
        You can still freely rotate from here with the mouse -- this only
        sets where the view *starts*. Pass the same values to
        :func:`plot_structure` to reproduce this exact starting view as a
        static figure once you've found an orientation you like.
    mark_atoms : dict or list of int, optional
        Atoms to label. A dict ``{index: label}`` uses custom text per
        atom; a list of indices labels each atom with its own index. Use
        e.g. ``[i for i, s in enumerate(atoms.get_chemical_symbols()) if s == 'O']``
        to label every oxygen. ``None`` (default) adds no labels.
    mark_bonds : list of (int or str, int or str), optional
        Extra atom pairs to connect with a highlighted dashed cylinder,
        independent of whatever 3Dmol.js draws automatically. Each
        endpoint is an atom index, or -- when *mark_atoms* is a dict --
        one of its label strings, e.g. with
        ``mark_atoms={0: 'O3', 3: 'O1'}`` you can pass
        ``mark_bonds=[('O3', 'O1')]`` instead of ``[(0, 3)]``.
        ``None`` (default) draws no extra bonds.
    bonds : bool
        If True, render ordinary stick bonds (3Dmol.js's own detection)
        alongside the atom spheres. ``False`` by default -- spheres only;
        pair with ``radius='vdw'`` for a cohesive-looking molecule rather
        than disconnected covalent-radius circles.
    bond_radius : float
        Radius (Å) of ordinary stick bonds.
    sphere_scale : float
        Multiplier applied to the chosen atomic radius.
    radius : {'covalent', 'vdw'}
        Which radius to size atoms by -- see
        :func:`chempiler.colours.elem_radius`. Use ``'vdw'`` for a
        space-filling look, particularly when ``bonds=False`` (with
        covalent radii and no bonds, atoms that are actually
        touching/bonded end up visibly disconnected).
    show_orientation : bool
        If True, overlay the live camera state (position + rotation
        quaternion, from 3Dmol.js's own ``getView()``) as text in the
        bottom-right corner, updated a few times a second as you drag to
        rotate. **Experimental**: there is no supported way to read
        3Dmol.js's live camera state back into Python, so this works by
        injecting raw JavaScript directly into the viewer's output rather
        than through py3Dmol's normal (JSON-only) command API -- it could
        not be rendered/verified without a live browser, so treat it as
        best-effort and check that it actually displays correctly. It
        shows 3Dmol.js's own raw view array, not the ``elev``/``azim``
        convention used here and in :func:`plot_structure` -- there's no
        conversion between the two yet.
    width, height : int
        Viewer size in pixels.

    Returns
    -------
    py3Dmol.view
        Return it as the last expression in a Jupyter cell to render
        inline. Drag to rotate, scroll to zoom -- native 3Dmol.js controls,
        no extra setup needed.
    """
    import py3Dmol
    from io import StringIO
    from ase.io import write
    from .colours import elem_color, elem_radius

    pos = _oriented_positions(atoms.get_positions(), elev=elev, azim=azim, roll=roll)
    atoms = atoms.copy()
    atoms.set_positions(pos)

    buf = StringIO()
    write(buf, atoms, format='xyz')

    view = py3Dmol.view(width=width, height=height)
    view.addModel(buf.getvalue(), 'xyz')

    for sym in set(atoms.get_chemical_symbols()):
        color = elem_color(sym)
        r = elem_radius(sym, kind=radius) * sphere_scale
        sym_style = {'sphere': {'color': color, 'radius': r}}
        if bonds:
            sym_style['stick'] = {'color': color, 'radius': bond_radius}
        view.setStyle({'elem': sym}, sym_style)

    if mark_atoms is not None:
        items = mark_atoms.items() if isinstance(mark_atoms, dict) else (
            (i, str(i)) for i in mark_atoms
        )
        for i, label in items:
            x, y, z = (float(v) for v in pos[i])
            view.addLabel(str(label), {
                'position': {'x': x, 'y': y, 'z': z},
                'backgroundColor': 'black',
                'backgroundOpacity': 0.7,
                'fontColor': 'white',
                'fontSize': 12,
                'showBackground': True,
            })

    if mark_bonds is not None:
        for i, j in mark_bonds:
            i = _resolve_atom_ref(i, mark_atoms)
            j = _resolve_atom_ref(j, mark_atoms)
            xi, yi, zi = (float(v) for v in pos[i])
            xj, yj, zj = (float(v) for v in pos[j])
            view.addCylinder({
                'start': {'x': xi, 'y': yi, 'z': zi},
                'end':   {'x': xj, 'y': yj, 'z': zj},
                'radius': bond_radius * 0.75,
                'color': '#444444',
                'dashed': True,
            })

    view.zoomTo()

    if show_orientation:
        # Experimental: py3Dmol's normal API (the __getattr__-based method
        # calls above) always JSON-encodes its arguments, so there's no way
        # to pass it a live JS callback through that path. Instead we
        # append raw JavaScript directly onto view.startjs -- a public
        # string attribute py3Dmol itself builds the same way internally
        # (confirmed by reading its source). Appending to updatejs instead
        # (an earlier version of this) was a bug: _make_html(), which is
        # what actually renders when the view is returned from a cell,
        # builds the displayed HTML from startjs+endjs only and clears
        # updatejs first -- so anything appended there was silently
        # discarded before ever reaching the browser. 'UNIQUEID' is a
        # literal placeholder py3Dmol substitutes with the real per-viewer
        # id at render time, same as every other command it generates; by
        # this point in startjs, viewer_UNIQUEID is already assigned and
        # the container div already exists, so both are safe to reference.
        # elev/azim/roll are recovered from 3Dmol.js's rotation quaternion
        # by converting to a matrix and reading it off exactly the way
        # _oriented_positions builds R = Rz(roll) . Rx(elev) . Rz(azim) --
        # verified by round-tripping this same math in Python (angles ->
        # matrix -> recovered angles -> reconstructed matrix, matching to
        # ~1e-16) over 2000 random trials before writing it here. (An
        # earlier elev/azim-only version of this was wrong: free mouse
        # rotation is a full 3-DOF orientation, and 2 numbers can't
        # represent that -- confirmed against a live reading that showed a
        # consistent mismatch until roll was added.) What is still NOT
        # verified (no way to, without a live browser): whether 3Dmol.js's
        # getView() array is really ordered [x, y, z, zoom, qx, qy, qz,
        # qw], and whether its quaternion represents the rotation in the
        # same sense mine does or its inverse (camera vs. object
        # rotation). If plot_structure(elev=.., azim=.., roll=..) still
        # doesn't reproduce what's shown here, that's the likely
        # remaining culprit. This reading is only meaningful if you called
        # view3d with elev=azim=roll=0 (the default) and did all rotation
        # by dragging -- it reports the drag rotation on top of whatever
        # was already baked into the coordinates, not the combined total.
        view.startjs += '''
(function() {
    var container = document.getElementById("3dmolviewer_UNIQUEID");
    if (!container) { return; }
    var overlay = document.createElement("div");
    overlay.style.position = "absolute";
    overlay.style.bottom = "4px";
    overlay.style.right = "4px";
    overlay.style.background = "rgba(0,0,0,0.6)";
    overlay.style.color = "white";
    overlay.style.font = "11px monospace";
    overlay.style.padding = "4px 7px";
    overlay.style.borderRadius = "3px";
    overlay.style.pointerEvents = "none";
    overlay.style.whiteSpace = "pre";
    overlay.style.textAlign = "left";
    container.appendChild(overlay);

    function quatToAngles(qx, qy, qz, qw) {
        var r02 = 2*(qx*qz + qy*qw);
        var r12 = 2*(qy*qz - qx*qw);
        var r20 = 2*(qx*qz - qy*qw);
        var r21 = 2*(qy*qz + qx*qw);
        var r22 = 1 - 2*(qx*qx + qy*qy);
        var el = Math.atan2(Math.hypot(r20, r21), r22) * 180 / Math.PI;
        var az = Math.atan2(r20, r21) * 180 / Math.PI;
        var ro = Math.atan2(r02, -r12) * 180 / Math.PI;
        return [el, az, ro];
    }

    function update() {
        var v = viewer_UNIQUEID.getView();
        var a = quatToAngles(v[4], v[5], v[6], v[7]);
        var lines = [
            "elev=" + a[0].toFixed(1) + "  azim=" + a[1].toFixed(1) + "  roll=" + a[2].toFixed(1),
            "zoom=" + v[3].toFixed(2),
        ];
        overlay.textContent = lines.join("\\n");
    }
    update();
    setInterval(update, 200);
})();
'''

    return view


def plot_structure(atoms, elev=0.0, azim=0.0, roll=0.0, mark_atoms=None, mark_bonds=None,
                   bonds=False, bond_scale=1.0, bond_color='#555555', bond_width=1.5,
                   mark_bond_color='#444444', mark_bond_width=1.5,
                   mark_bond_style=(0, (4, 3)),
                   sphere_scale=1.0, radius='covalent',
                   focus_atoms=None, fade_alpha=0.25,
                   ax=None, figsize=(5, 5)):
    """Static 3D-projected structure figure, styled to match chempiler's RDF insets.

    The counterpart to :func:`view3d` for figures that need to render
    without JavaScript (e.g. embedded in a notebook viewed on GitHub,
    which strips interactive JS output -- see :func:`view3d`'s docs).
    Passing the same *elev*/*azim*/*roll* you settled on interactively in
    :func:`view3d` reproduces that exact starting orientation here, since
    both rotate the atom coordinates with the same
    :func:`_oriented_positions` before rendering. All three matter: free
    mouse rotation has 3 rotational degrees of freedom, and dropping
    *roll* reproduces the wrong orientation for anything but a pure
    elevation+azimuth change.

    Atoms are drawn with :func:`draw_atom_2d` -- the same shaded-circle
    style already used for the RDF-inset structures -- rather than
    matplotlib's native (and rather poor) 3D sphere rendering: the
    structure is orthographically projected to 2D in Python first (simply
    dropping the rotated z-coordinate, which is used only to depth-sort
    atoms and bonds back-to-front via ``zorder`` -- a painter's algorithm,
    not true occlusion, but adequate for the compact clusters this is
    meant for).

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to draw.
    elev, azim, roll : float
        Orientation in degrees -- see :func:`view3d`.
    mark_atoms : dict or list of int, optional
        Atoms to label, drawn above everything else. Same convention as
        :func:`view3d`.
    mark_bonds : list of (int, int), optional
        Extra atom-index pairs to connect with a highlighted dashed line,
        independent of the automatically detected covalent bonds. Same
        convention as :func:`view3d`.
    bonds : bool
        If True, detect and draw ordinary covalent bonds (see
        :func:`chempiler.perception.get_bonds`). ``False`` by default --
        spheres only; pair with ``radius='vdw'`` for a cohesive-looking
        molecule rather than disconnected covalent-radius circles.
    bond_scale : float
        Scale factor passed to :func:`chempiler.perception.get_bonds`.
    bond_color : str
        Line colour for ordinary bonds.
    bond_width : float
        Line width for ordinary bonds.
    mark_bond_color : str
        Line colour for ``mark_bonds`` lines.
    mark_bond_width : float
        Line width for ``mark_bonds`` lines.
    mark_bond_style : matplotlib linestyle
        Line style for ``mark_bonds`` lines, e.g. ``'-'`` for solid instead
        of the default dashed ``(0, (4, 3))``.
    sphere_scale : float
        Multiplier applied to the chosen atomic radius (passed to
        :func:`draw_atom_2d`).
    radius : {'covalent', 'vdw'}
        Which radius to size atoms by -- see :func:`elem_radius`. Use
        ``'vdw'`` for a space-filling look, particularly when
        ``bonds=False`` (with covalent radii and no bonds, atoms that are
        actually touching/bonded end up visibly disconnected).
    focus_atoms : list of int or str, optional
        Atoms to keep fully opaque; every other atom, and every ordinary
        bond not between two focus atoms, is drawn at *fade_alpha*.
        Entries may be indices or *mark_atoms* labels (e.g. ``'O*'``).
        ``None`` (default) draws everything opaque.
    fade_alpha : float
        Opacity of the atoms and bonds outside *focus_atoms*.
    ax : matplotlib Axes, optional
        Draw into this axes instead of creating a new figure.
    figsize : tuple
        Figure size in inches, if *ax* is not given.

    Returns
    -------
    fig, ax : matplotlib Figure, Axes
        ``fig`` is ``None`` if *ax* was supplied. Call
        ``fig.savefig(...)`` for a plain PNG/SVG that renders everywhere,
        including GitHub's notebook preview.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    from .colours import draw_atom_2d, elem_radius
    from .perception import get_bonds

    syms = atoms.get_chemical_symbols()
    rotated = _oriented_positions(atoms.get_positions(), elev=elev, azim=azim, roll=roll)
    xy = rotated[:, :2]
    depth = rotated[:, 2]

    bond_pairs = list(get_bonds(atoms, bond_scale=bond_scale)) if bonds else []
    resolved_mark_bonds = []
    if mark_bonds is not None:
        for i, j in mark_bonds:
            resolved_mark_bonds.append((
                _resolve_atom_ref(i, mark_atoms),
                _resolve_atom_ref(j, mark_atoms),
            ))

    # Painter's algorithm: draw back-to-front, so nearer items end up on
    # top. Atoms and bonds are depth-sorted together in one combined pass
    # -- each bond's depth is its own two-endpoint average, not "pinned"
    # to either endpoint -- so a bond spanning two atoms at very different
    # depths (e.g. a mark_bonds contact between molecules, unlike an
    # ordinary short covalent bond where both ends are nearly the same
    # depth) still slots into its own correct position in the stack
    # instead of being forced entirely behind or in front of atoms it
    # should visually pass between.
    if focus_atoms is None:
        focus = set(range(len(atoms)))
    else:
        focus = {_resolve_atom_ref(ref, mark_atoms) for ref in focus_atoms}

    draw_items = [(depth[i], 'atom', i) for i in range(len(atoms))]
    draw_items += [((depth[i] + depth[j]) / 2.0, 'bond', (i, j)) for i, j in bond_pairs]
    draw_items += [((depth[i] + depth[j]) / 2.0, 'mark_bond', (i, j)) for i, j in resolved_mark_bonds]
    draw_items.sort(key=lambda t: t[0])

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # zorder_base=2 matches draw_atom_2d's own default so this composes
    # with anything the caller adds at a lower zorder.
    for rank, (_, kind, payload) in enumerate(draw_items):
        z = 2 + rank
        if kind == 'atom':
            i = payload
            draw_atom_2d(ax, xy[i, 0], xy[i, 1], syms[i], scale=sphere_scale,
                         radius=radius, zorder_base=z,
                         alpha=1.0 if i in focus else fade_alpha)
        elif kind == 'bond':
            i, j = payload
            ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=bond_color, lw=bond_width,
                    solid_capstyle='round', zorder=z,
                    alpha=1.0 if (i in focus and j in focus) else fade_alpha)
        else:  # mark_bond
            i, j = payload
            ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=mark_bond_color, lw=mark_bond_width,
                    ls=mark_bond_style, solid_capstyle='round', zorder=z)

    if mark_atoms is not None:
        items = mark_atoms.items() if isinstance(mark_atoms, dict) else (
            (i, str(i)) for i in mark_atoms
        )
        top = 3 + len(draw_items)
        for i, label in items:
            ax.text(xy[i, 0], xy[i, 1], str(label), ha='center', va='center',
                    fontsize=9, color='white', zorder=top,
                    bbox=dict(boxstyle='round,pad=0.15', fc='black',
                              ec='none', alpha=0.7))

    # add_patch()'d Circles (draw_atom_2d) don't participate in matplotlib's
    # autoscaling the way plot()/text() do, so without explicit limits the
    # view fits only whatever text labels happen to be present -- cropping
    # out most or all of the actual structure. Pad by the largest atom
    # radius actually in play so every circle is fully inside the frame.
    pad = max(elem_radius(s, kind=radius) for s in syms) * sphere_scale * 1.15
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig, ax
