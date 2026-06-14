"""Generic state-transition tracker for trajectory analysis.

Decoupled from chemistry: the caller supplies an entity_fn (which atoms or
molecules to watch) and a state_fn (what state each entity is in). Tracker
records transitions and supports a persistence filter to suppress transient
noise near transition boundaries.
"""

import numpy as np


class Tracker:
    """
    Generic state-transition tracker.

    Calls state_fn(frame, entity) each frame for each entity and records
    a transition whenever the returned state changes. The persistence
    parameter requires the new state to hold for N consecutive frames
    before the transition is committed (suppresses single-frame noise).
    """

    def __init__(self, frames):
        self.frames = frames
        self.n_frames = len(frames)

    def track(self, entity_fn, state_fn, persistence=1):
        """
        Parameters
        ----------
        entity_fn : callable(frames) -> iterable of int
            Returns the atom/entity indices to track.
        state_fn : callable(frame, entity) -> hashable
            Returns the current state of an entity in a given frame.
        persistence : int
            Frames the new state must persist before a transition is recorded.

        Returns
        -------
        dict with keys:
            transitions     : list of (frame_idx, entity, from_state, to_state)
            n_transitions   : int
            residence_times : ndarray — frames in final state at end of trajectory
        """
        entities = np.asarray(entity_fn(self.frames), dtype=np.int32)

        if len(entities) == 0:
            return {"n_transitions": 0, "transitions": [], "residence_times": np.array([])}

        prev = {e: state_fn(self.frames[0], e) for e in entities}
        pending = {e: None for e in entities}
        pending_count = {e: 0 for e in entities}
        residence = {e: 0 for e in entities}
        transitions = []

        for t in range(1, self.n_frames):
            frame = self.frames[t]
            for e in entities:
                s = state_fn(frame, e)
                old = prev[e]

                if s != old:
                    if pending[e] == s:
                        pending_count[e] += 1
                    else:
                        pending[e] = s
                        pending_count[e] = 1

                    if pending_count[e] >= persistence:
                        transitions.append((t, int(e), old, s))
                        prev[e] = s
                        pending[e] = None
                        pending_count[e] = 0
                        residence[e] = 0
                else:
                    pending[e] = None
                    pending_count[e] = 0
                    residence[e] += 1

        return {
            "transitions": transitions,
            "n_transitions": len(transitions),
            "residence_times": np.array(list(residence.values())),
        }
