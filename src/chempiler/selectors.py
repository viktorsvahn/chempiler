"""Atom and molecule index selectors.

All functions return NumPy int32 arrays of indices into the atom or molecule
lists of a Frame. These are the building blocks for entity_fn and state_fn
callables used by Tracker and the analysis functions in state_engine.
"""

import numpy as np


def atoms(frame, symbol=None):
    """Return atom indices, optionally filtered by element symbol.

    Parameters
    ----------
    frame : Frame
    symbol : str, optional
        Element symbol (e.g. ``"O"``). If None, returns all atom indices.

    Returns
    -------
    numpy.ndarray of int32
    """
    if symbol is None:
        return np.arange(len(frame.symbols), dtype=np.int32)
    return np.array(
        [i for i, s in enumerate(frame.symbols) if s == symbol],
        dtype=np.int32,
    )


def molecules(frame):
    """Return all molecule indices for a frame.

    Parameters
    ----------
    frame : Frame

    Returns
    -------
    numpy.ndarray of int32
    """
    return np.arange(len(frame.molecules), dtype=np.int32)


def formulas(frame, formula_list=None):
    """Return molecule indices matching a list of molecular formulas.

    Parameters
    ----------
    frame : Frame
    formula_list : list of str, optional
        Molecular formulas to select (e.g. ``["H2O", "HO"]``).
        If None, returns all molecule indices.

    Returns
    -------
    numpy.ndarray of int32
    """
    if formula_list is None:
        return molecules(frame)
    idx = []
    for f in formula_list:
        idx.extend(frame.formula_to_mols.get(f, []))
    return np.array(idx, dtype=np.int32)


def resolve(frame, selector):
    """Resolve a selector to an array of atom indices.

    Used by rdf() to accept flexible center/target specifications.

    Parameters
    ----------
    frame : Frame
    selector : str or dict
        - ``"O"``           — all O atom indices.
        - ``{"H2O": "H"}``  — H atom indices inside H2O molecules.
        - ``{"HO": None}``  — all atom indices inside HO molecules.

    Returns
    -------
    numpy.ndarray of int32

    Raises
    ------
    ValueError
        If the selector type is not recognised.
    """
    if isinstance(selector, str):
        return atoms(frame, selector)

    if isinstance(selector, dict):
        formula, element = next(iter(selector.items()))
        mol_ids = frame.formula_to_mols.get(formula, [])
        idx = [
            a
            for mid in mol_ids
            for a in frame.molecules[mid]
            if element is None or frame.symbols[a] == element
        ]
        return np.array(idx, dtype=np.int32)

    raise ValueError(f"Unknown selector: {selector!r}")
