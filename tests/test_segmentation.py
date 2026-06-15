"""Tests for trajectory segmentation functions.

Unit tests use minimal synthetic Frames. Integration tests load the real
39-water trajectory and check that results are structurally sound.
"""

import numpy as np
import pytest
from pathlib import Path
from ase import Atoms

from chempiler.frame import Frame
from chempiler.segmentation import segment_by_molecule_count, lifetime_segments

TESTS_DIR = Path(__file__).parent


# ============================================================
# helpers
# ============================================================

def make_frame(symbols, positions, molecules=None, cell=(100.0, 100.0, 100.0)):
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=False)
    if molecules is None:
        molecules = [[i] for i in range(len(symbols))]
    frame = Frame(atoms=atoms, molecules=molecules)
    frame.build()
    return frame


def h2o_frame():
    return make_frame(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]],
        molecules=[[0, 1, 2]],
    )


def ho_frame():
    return make_frame(
        ["O", "H"],
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]],
        molecules=[[0, 1]],
    )


def mixed_h2o_ho_frame():
    """Frame containing one H2O and one HO molecule."""
    return make_frame(
        ["O", "H", "H", "O", "H"],
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0],
         [5.0, 0.0, 0.0], [5.96, 0.0, 0.0]],
        molecules=[[0, 1, 2], [3, 4]],
    )


# ============================================================
# segment_by_molecule_count — unit tests
# ============================================================

class TestSegmentByMoleculeCount:

    def test_stable_trajectory_is_single_segment(self):
        frames = [h2o_frame()] * 200
        segs = segment_by_molecule_count(frames, block=50, threshold=0.5)
        assert len(segs) == 1
        assert segs[0] == (0, 200)

    def test_segments_cover_full_trajectory(self):
        frames = [h2o_frame()] * 100 + [mixed_h2o_ho_frame()] * 100
        segs = segment_by_molecule_count(frames, block=50, threshold=0.5)
        assert segs[0][0] == 0
        assert segs[-1][1] == 200
        # Segments must be contiguous
        for i in range(1, len(segs)):
            assert segs[i][0] == segs[i - 1][1]

    def test_step_change_detected(self):
        # H2O has 1 molecule; mixed has 2 — clear step at frame 100
        frames = [h2o_frame()] * 100 + [mixed_h2o_ho_frame()] * 100
        segs = segment_by_molecule_count(frames, block=50, threshold=0.5)
        assert len(segs) == 2

    def test_threshold_prevents_spurious_split(self):
        # All frames have 1 molecule — a threshold of 0.5 should never split them
        frames = [h2o_frame()] * 200
        segs = segment_by_molecule_count(frames, block=50, threshold=0.5)
        assert len(segs) == 1

    def test_returns_list_of_tuples(self):
        segs = segment_by_molecule_count([h2o_frame()] * 100)
        assert isinstance(segs, list)
        for s in segs:
            assert isinstance(s, tuple) and len(s) == 2


# ============================================================
# lifetime_segments — unit tests
# ============================================================

class TestLifetimeSegments:

    def test_formula_never_present_returns_empty(self):
        frames = [h2o_frame()] * 10
        assert lifetime_segments(frames, "HO") == []

    def test_formula_always_present_returns_one_segment(self):
        frames = [h2o_frame()] * 10
        segs = lifetime_segments(frames, "H2O")
        assert segs == [(0, 10)]

    def test_single_occurrence_in_middle(self):
        frames = [h2o_frame(), h2o_frame(), mixed_h2o_ho_frame(), h2o_frame(), h2o_frame()]
        segs = lifetime_segments(frames, "HO")
        assert segs == [(2, 3)]

    def test_two_disconnected_segments(self):
        frames = (
            [h2o_frame()] * 3
            + [mixed_h2o_ho_frame()] * 2
            + [h2o_frame()] * 3
            + [mixed_h2o_ho_frame()] * 2
        )
        segs = lifetime_segments(frames, "HO")
        assert len(segs) == 2
        assert segs[0] == (3, 5)
        assert segs[1] == (8, 10)

    def test_segment_starts_at_frame_zero(self):
        frames = [mixed_h2o_ho_frame()] * 3 + [h2o_frame()] * 3
        segs = lifetime_segments(frames, "HO")
        assert segs[0][0] == 0

    def test_segment_ends_at_last_frame(self):
        frames = [h2o_frame()] * 3 + [mixed_h2o_ho_frame()] * 3
        segs = lifetime_segments(frames, "HO")
        assert segs[-1][1] == 6

    def test_single_frame_segment(self):
        frames = [h2o_frame(), mixed_h2o_ho_frame(), h2o_frame()]
        segs = lifetime_segments(frames, "HO")
        assert segs == [(1, 2)]

    def test_intervals_are_half_open(self):
        frames = [mixed_h2o_ho_frame()] * 5
        segs = lifetime_segments(frames, "HO")
        start, end = segs[0]
        assert end - start == 5

    def test_returns_list_of_tuples(self):
        frames = [h2o_frame()] * 5
        segs = lifetime_segments(frames, "H2O")
        assert isinstance(segs, list)
        for s in segs:
            assert isinstance(s, tuple) and len(s) == 2

    def test_empty_frames_returns_empty(self):
        assert lifetime_segments([], "H2O") == []

    def test_frame_slice_is_valid(self):
        frames = [h2o_frame()] * 3 + [mixed_h2o_ho_frame()] * 4 + [h2o_frame()] * 3
        segs = lifetime_segments(frames, "HO")
        start, end = segs[0]
        sliced = frames[start:end]
        assert len(sliced) == end - start
        assert all("HO" in f.formula_to_mols for f in sliced)


# ============================================================
# integration — real 39-water trajectory
# ============================================================

@pytest.fixture(scope="module")
def water_frames():
    from chempiler import ChempilerTrajectory
    t = ChempilerTrajectory(
        str(TESTS_DIR / "39_water_OH.traj"),
        mode="molecular",
        covalent_scale=1.0,
    )
    t.build(cache_file=str(TESTS_DIR / "cache_file.h5"))
    return t.frames


class TestLifetimeSegmentsIntegration:

    def test_h2o_is_one_contiguous_segment(self, water_frames):
        segs = lifetime_segments(water_frames, "H2O")
        assert len(segs) == 1
        assert segs[0] == (0, len(water_frames))

    def test_transient_species_has_multiple_short_segments(self, water_frames):
        segs = lifetime_segments(water_frames, "HO")
        assert len(segs) > 1
        # Transient species — segments are short
        lengths = [e - s for s, e in segs]
        assert max(lengths) < len(water_frames)

    def test_segments_non_overlapping_and_ordered(self, water_frames):
        segs = lifetime_segments(water_frames, "HO")
        for i in range(1, len(segs)):
            assert segs[i][0] >= segs[i - 1][1]

    def test_all_frames_in_segment_contain_formula(self, water_frames):
        segs = lifetime_segments(water_frames, "HO")
        for start, end in segs:
            for frame in water_frames[start:end]:
                assert "HO" in frame.formula_to_mols

    def test_frames_outside_segment_lack_formula(self, water_frames):
        segs = lifetime_segments(water_frames, "HO")
        in_seg = set()
        for start, end in segs:
            in_seg.update(range(start, end))
        outside = [i for i in range(len(water_frames)) if i not in in_seg]
        for i in outside:
            assert "HO" not in water_frames[i].formula_to_mols

    def test_unknown_formula_returns_empty(self, water_frames):
        assert lifetime_segments(water_frames, "He") == []

    def test_traj_method_matches_function(self, water_frames):
        from chempiler import ChempilerTrajectory
        from chempiler.segmentation import lifetime_segments as fn
        t = ChempilerTrajectory(
            str(TESTS_DIR / "39_water_OH.traj"),
            mode="molecular",
            covalent_scale=1.0,
        )
        t.build(cache_file=str(TESTS_DIR / "cache_file.h5"))
        assert t.lifetime_segments("HO") == fn(water_frames, "HO")
