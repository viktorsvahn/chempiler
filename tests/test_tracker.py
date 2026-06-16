"""Tests for the generic Tracker class (chempiler.core.tracker)."""

import numpy as np
import pytest

from chempiler.core.tracker import Tracker


# ============================================================
# helpers — lightweight fake frames
# ============================================================

def make_tracker(state_sequence):
    """Return (Tracker, entity_fn, state_fn) for a single entity whose state
    follows *state_sequence* frame-by-frame.

    Frames are just integer indices so that state_fn can do a simple lookup.
    """
    frames = list(range(len(state_sequence)))
    entity_fn = lambda frames_list: [0]
    state_fn = lambda frame, entity: state_sequence[frame]
    return Tracker(frames), entity_fn, state_fn


def two_entity_tracker(seq0, seq1):
    """Two entities with independent state sequences (must be same length)."""
    assert len(seq0) == len(seq1)
    frames = list(range(len(seq0)))
    entity_fn = lambda frames_list: [0, 1]
    state_fn = lambda frame, entity: (seq0 if entity == 0 else seq1)[frame]
    return Tracker(frames), entity_fn, state_fn


# ============================================================
# no transitions
# ============================================================

class TestTrackerNoTransitions:

    def test_stable_state_no_transitions(self):
        t, ef, sf = make_tracker(["A"] * 10)
        result = t.track(ef, sf)
        assert result["n_transitions"] == 0
        assert result["transitions"] == []

    def test_single_frame_no_transitions(self):
        t, ef, sf = make_tracker(["A"])
        result = t.track(ef, sf)
        assert result["n_transitions"] == 0

    def test_empty_entities_empty_result(self):
        frames = list(range(5))
        t = Tracker(frames)
        result = t.track(lambda _: [], lambda f, e: None)
        assert result["n_transitions"] == 0
        assert result["transitions"] == []
        assert len(result["residence_times"]) == 0


# ============================================================
# single transition
# ============================================================

class TestTrackerSingleTransition:

    def test_one_transition_detected(self):
        t, ef, sf = make_tracker(["A", "A", "A", "B", "B", "B"])
        result = t.track(ef, sf)
        assert result["n_transitions"] == 1

    def test_transition_frame_index_correct(self):
        t, ef, sf = make_tracker(["A", "A", "B", "B"])
        result = t.track(ef, sf)
        frame_idx, _, _, _ = result["transitions"][0]
        assert frame_idx == 2

    def test_transition_from_and_to_correct(self):
        t, ef, sf = make_tracker(["A", "B", "B"])
        result = t.track(ef, sf)
        _, entity, from_state, to_state = result["transitions"][0]
        assert from_state == "A"
        assert to_state == "B"

    def test_entity_index_in_transition(self):
        t, ef, sf = make_tracker(["X", "Y", "Y"])
        result = t.track(ef, sf)
        _, entity, _, _ = result["transitions"][0]
        assert entity == 0

    def test_n_transitions_matches_list_length(self):
        t, ef, sf = make_tracker(["A", "B", "A", "B"])
        result = t.track(ef, sf, persistence=1)
        assert result["n_transitions"] == len(result["transitions"])


# ============================================================
# persistence filter
# ============================================================

class TestTrackerPersistence:

    def test_persistence_2_suppresses_one_frame_flicker(self):
        # A A B A A — flicker at frame 2, back to A immediately
        t, ef, sf = make_tracker(["A", "A", "B", "A", "A"])
        result = t.track(ef, sf, persistence=2)
        assert result["n_transitions"] == 0

    def test_persistence_1_counts_flicker(self):
        t, ef, sf = make_tracker(["A", "B", "A"])
        result = t.track(ef, sf, persistence=1)
        assert result["n_transitions"] == 2

    def test_persistence_2_allows_sustained_change(self):
        # New state held for 3 frames → transition committed at frame 3
        t, ef, sf = make_tracker(["A", "A", "B", "B", "B"])
        result = t.track(ef, sf, persistence=2)
        assert result["n_transitions"] == 1

    def test_transition_recorded_at_correct_frame_with_persistence(self):
        # A A B B B — persistence=2: transition recorded when B has held for 2 frames
        t, ef, sf = make_tracker(["A", "A", "B", "B", "B"])
        result = t.track(ef, sf, persistence=2)
        frame_idx, _, _, _ = result["transitions"][0]
        # Transition at frame 3 (second frame of B)
        assert frame_idx == 3

    def test_persistence_higher_than_change_duration_suppresses(self):
        # B lasts only 2 frames, persistence=3
        t, ef, sf = make_tracker(["A", "B", "B", "A", "A"])
        result = t.track(ef, sf, persistence=3)
        assert result["n_transitions"] == 0


# ============================================================
# multiple entities
# ============================================================

class TestTrackerMultipleEntities:

    def test_two_independent_entities(self):
        t, ef, sf = two_entity_tracker(
            ["A", "B", "B"],  # entity 0 hops at frame 1
            ["X", "X", "X"],  # entity 1 stays
        )
        result = t.track(ef, sf)
        assert result["n_transitions"] == 1
        _, entity, _, _ = result["transitions"][0]
        assert entity == 0

    def test_both_entities_hop_simultaneously(self):
        t, ef, sf = two_entity_tracker(
            ["A", "B", "B"],
            ["X", "Y", "Y"],
        )
        result = t.track(ef, sf)
        assert result["n_transitions"] == 2


# ============================================================
# residence_times
# ============================================================

class TestTrackerResidenceTimes:

    def test_residence_times_non_negative(self):
        t, ef, sf = make_tracker(["A", "A", "B", "B", "B"])
        result = t.track(ef, sf)
        assert (result["residence_times"] >= 0).all()

    def test_stable_trajectory_has_large_residence(self):
        n = 20
        t, ef, sf = make_tracker(["A"] * n)
        result = t.track(ef, sf)
        assert result["residence_times"][0] == n - 1

    def test_residence_times_length_equals_n_entities(self):
        t, ef, sf = make_tracker(["A", "B", "B"])
        result = t.track(ef, sf)
        assert len(result["residence_times"]) == 1  # one entity

    def test_output_keys_present(self):
        t, ef, sf = make_tracker(["A", "B"])
        result = t.track(ef, sf)
        assert {"transitions", "n_transitions", "residence_times"} <= result.keys()
