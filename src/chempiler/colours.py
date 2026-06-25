"""Atom colour scheme and 2D drawing utilities shared across all chempiler visuals.

Colour philosophy
-----------------
Jmol hue semantics (element recognisability) with Paul Tol's perceptually
uniform, colourblind-safe values (https://personal.sron.nl/~pault/).
Orange for P is borrowed from Tol vibrant — Tol bright has no orange.
H is mapped to Tol grey rather than white so it is visible on white backgrounds.
"""

import numpy as np
from ase.data import covalent_radii as _ASE_COV_RADII, atomic_numbers as _ASE_ANUM

# ---------------------------------------------------------------------------
# Tol-ified Jmol colour scheme
# ---------------------------------------------------------------------------
ELEM_COLOR = {
    'H':  '#BBBBBB',  # Tol grey    (Jmol: white — invisible on white bg)
    'C':  '#555555',  # dark grey   (Jmol: #909090 — darkened for contrast)
    'N':  '#4477AA',  # Tol blue
    'O':  '#EE6677',  # Tol red
    'F':  '#66CCEE',  # Tol cyan    (distinct from Cl green)
    'S':  '#CCBB44',  # Tol yellow  (Jmol: bright yellow — weak on white bg)
    'Cl': '#228833',  # Tol green
    'P':  '#EE7733',  # Tol vibrant orange
    'Be': '#C2FF00',
    'Na': '#AB5CF2',
    'Mg': '#8AFF00',
    'Al': '#BFA6A6',
    'Si': '#F0C8A0',
    'Br': '#A62929',
    'I':  '#940094',
}

_FALLBACK_COLOR = '#BBBBBB'


def elem_color(symbol):
    """Return the Tol-ified Jmol colour for *symbol*, falling back to grey."""
    return ELEM_COLOR.get(symbol, _FALLBACK_COLOR)


def elem_radius(symbol):
    """Return the ASE covalent radius (Å) for *symbol*, falling back to 0.7 Å."""
    n = _ASE_ANUM.get(symbol)
    if n is None:
        return 0.70
    return float(_ASE_COV_RADII[n])


# ---------------------------------------------------------------------------
# Atom drawing
# ---------------------------------------------------------------------------
def draw_atom_2d(ax, x, y, symbol, scale=1.0, zorder_base=2):
    """Draw a single atom as a shaded circle on *ax*.

    The circle uses the Tol-ified Jmol colour with a dark outline and a small
    white specular highlight offset to the upper-left — matching the style used
    in :func:`chempiler.rdf.plot_rdf` insets.

    Parameters
    ----------
    ax : matplotlib Axes
    x, y : float
        Centre position in data coordinates (Å).
    symbol : str
        Element symbol, e.g. ``'O'``.
    scale : float
        Multiplier applied to the covalent radius.
    zorder_base : int
        z-order for the atom body; the highlight is drawn at ``zorder_base + 1``.
    """
    from matplotlib.patches import Circle

    r = elem_radius(symbol) * scale
    fc = elem_color(symbol)

    ax.add_patch(Circle(
        (x, y), radius=r,
        color=fc, ec='#111111', lw=0.65, zorder=zorder_base,
    ))
    ax.add_patch(Circle(
        (x - 0.3 * r, y + 0.3 * r), radius=0.18 * r,
        color='white', alpha=0.6, lw=0, zorder=zorder_base + 1,
    ))


def draw_molecule_2d(ax, positions_2d, symbols, scale=1.0, zorder_base=2):
    """Draw a set of atoms from pre-projected 2D positions.

    Parameters
    ----------
    ax : matplotlib Axes
    positions_2d : array-like, shape (N, 2)
        2D coordinates in data units (Å) for each atom.
    symbols : list of str
        Element symbol for each atom.
    scale : float
        Multiplier applied to all covalent radii.
    zorder_base : int
        z-order for atom bodies; highlights are at ``zorder_base + 1``.
    """
    positions_2d = np.asarray(positions_2d)
    for (x, y), sym in zip(positions_2d, symbols):
        draw_atom_2d(ax, float(x), float(y), sym, scale=scale,
                     zorder_base=zorder_base)
