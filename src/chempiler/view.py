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
