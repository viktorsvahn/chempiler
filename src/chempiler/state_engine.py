"""High-level analysis functions for tracking chemical state changes.

Each function wraps the generic Tracker (or the per-frame track_state loop for
analyses where the entity set can change between frames) with chemistry-specific
entity and state definitions.

Functions
---------
atom_hop               Track mobile ions hopping between host atoms.
ligand_exchange        Track changes in molecular formula over time.
coordination_dynamics  Track changes in coordination environment over time.
"""

import numpy as np
from chempiler.core.tracker import Tracker
from chempiler.core.state_field import nearest_host
from chempiler.selectors import atoms, molecules


def _track_state(frames, entities_per_frame, state_fn, mode="generic"):
    """Generic per-frame state tracker for analyses with changing entity sets.

    Unlike Tracker.track, entity_fn is called on every frame, so entities that
    appear or disappear mid-trajectory (e.g., newly formed molecules) are
    handled correctly.

    Parameters
    ----------
    frames : list of Frame
    entities_per_frame : callable(frame) -> iterable of int
        Returns the entities to observe in a given frame.
    state_fn : callable(frame, entity) -> hashable
        Returns the current state of an entity.
    mode : str
        Label stored in the result dict.

    Returns
    -------
    dict with keys:
        mode            : str
        transitions     : list of (frame_idx, entity, from_state, to_state)
        n_transitions   : int
        residence_times : numpy.ndarray — frames spent in final state
    """
    transitions = []
    residence = {}
    prev_state = {}

    frame0 = frames[0]
    for e in entities_per_frame(frame0):
        prev_state[e] = state_fn(frame0, e)
        residence[e] = 1

    for t in range(1, len(frames)):
        frame = frames[t]
        for e in set(entities_per_frame(frame)):
            s = state_fn(frame, e)
            if e not in prev_state:
                prev_state[e] = s
                residence[e] = 1
                continue
            if s != prev_state[e]:
                transitions.append((t, e, prev_state[e], s))
                prev_state[e] = s
                residence[e] = 1
            else:
                residence[e] += 1

    return {
        "mode": mode,
        "n_transitions": len(transitions),
        "transitions": transitions,
        "residence_times": np.array(list(residence.values())),
    }


def atom_hop(frames, tracked="H", host="O", cutoff=1.25, persistence=1):
    """Track atoms of one species hopping between atoms of another species.

    The state of each tracked atom is the index of its nearest host atom within
    *cutoff*. A hop is recorded when this index changes. Works for any mobile
    species: proton transfer (H/O), lithium hopping (Li/O), etc.

    Parameters
    ----------
    frames : list of Frame
    tracked : str
        Element symbol of the mobile species (e.g. ``"H"``).
    host : str
        Element symbol of the host species (e.g. ``"O"``).
    cutoff : float
        Distance cutoff in Ångström for considering a bond.
    persistence : int
        Minimum consecutive frames the new state must hold before a transition
        is recorded. Increase to suppress sub-picosecond rattling.

    Returns
    -------
    dict with keys:
        transitions     : list of (frame_idx, tracked_atom_idx, from_host_idx, to_host_idx)
        n_transitions   : int
        residence_times : numpy.ndarray
    """
    engine = Tracker(frames)
    syms = frames[0].symbols

    def entity_fn(frames):
        return np.array([i for i, s in enumerate(syms) if s == tracked], dtype=np.int32)

    def state_fn(frame, atom_idx):
        return nearest_host(frame, atom_idx, host, cutoff)

    return engine.track(entity_fn, state_fn, persistence=persistence)


def ligand_exchange(frames, formulas=None):
    """Track changes in molecular formula for each molecule over time.

    A transition is recorded whenever a molecule's formula changes, which
    indicates a bond-breaking or bond-forming event (reaction).

    Parameters
    ----------
    frames : list of Frame
    formulas : list of str, optional
        Restrict tracking to molecules with these formulas. If None, all
        molecules are tracked.

    Returns
    -------
    dict with keys:
        mode            : "ligand_exchange"
        transitions     : list of (frame_idx, mol_idx, from_formula, to_formula)
        n_transitions   : int
        residence_times : numpy.ndarray
    """
    def entities(frame):
        if formulas is None:
            return molecules(frame)
        idx = []
        for f in formulas:
            idx.extend(frame.formula_to_mols.get(f, []))
        return np.array(idx, dtype=np.int32)

    def state_fn(frame, m):
        return frame.formulas[m]

    return _track_state(frames, entities, state_fn, mode="ligand_exchange")


def coordination_dynamics(frames, atom_symbol="O"):
    """Track changes in the coordination environment of each atom over time.

    The state of each atom is the sorted tuple of element symbols of all atoms
    in its molecule. A transition signals that the atom's bonding environment
    has changed.

    Parameters
    ----------
    frames : list of Frame
    atom_symbol : str
        Element to track (e.g. ``"O"`` to follow oxygen coordination).

    Returns
    -------
    dict with keys:
        mode            : "coordination"
        transitions     : list of (frame_idx, atom_idx, from_env, to_env)
        n_transitions   : int
        residence_times : numpy.ndarray
    """
    def entities(frame):
        return atoms(frame, atom_symbol)

    def state_fn(frame, a):
        mol_id = frame.atom_to_mol[a]
        if mol_id < 0:
            return None
        mol = frame.molecules[mol_id]
        return tuple(sorted(frame.atoms[i].symbol for i in mol))

    return _track_state(frames, entities, state_fn, mode="coordination")
