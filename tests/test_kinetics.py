"""Tests for chempiler.kinetics.reaction_kinetics.

Unit tests use synthetic frame sequences where the appearance/disappearance of a
formula is under full control, so event counts and lifetimes are known exactly.
Integration tests run against the real 39-water trajectory.
"""

import numpy as np
import pytest
from pathlib import Path
from ase import Atoms

from chempiler.frame import Frame
from chempiler.kinetics import reaction_kinetics

TESTS_DIR = Path(__file__).parent


# ============================================================
# helpers
# ============================================================

def make_frame(symbols, positions, molecules, cell=(100.0, 100.0, 100.0)):
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=False)
    frame = Frame(atoms=atoms, molecules=molecules)
    frame.build()
    return frame


def h2o_frame():
    return make_frame(["O", "H", "H"],
                      [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                      [[0, 1, 2]])


def ho_frame():
    return make_frame(["O", "H"],
                      [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
                      [[0, 1]])


# ============================================================
# output structure
# ============================================================

class TestKineticsOutputStructure:

    def test_returns_dict(self):
        frames = [ho_frame()] * 5
        result = reaction_kinetics(frames, "HO")
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        k = reaction_kinetics([ho_frame()] * 5, "HO")
        for key in ("n_events", "lifetimes", "mean_lifetime", "median_lifetime", "rate"):
            assert key in k

    def test_lifetimes_is_ndarray(self):
        k = reaction_kinetics([ho_frame()] * 5, "HO")
        assert isinstance(k["lifetimes"], np.ndarray)

    def test_n_events_is_int(self):
        k = reaction_kinetics([ho_frame()] * 5, "HO")
        assert isinstance(k["n_events"], int)


# ============================================================
# event counting
# ============================================================

class TestKineticsEventCounting:

    def test_single_contiguous_block_is_one_event(self):
        # HO present for frames 0-4: one event
        k = reaction_kinetics([ho_frame()] * 5, "HO")
        assert k["n_events"] == 1

    def test_two_separated_blocks_are_two_events(self):
        # HO in 0-2, absent in 3-5, HO in 6-8 → 2 events
        frames = [ho_frame()] * 3 + [h2o_frame()] * 3 + [ho_frame()] * 3
        k = reaction_kinetics(frames, "HO")
        assert k["n_events"] == 2

    def test_formula_absent_gives_zero_events(self):
        k = reaction_kinetics([h2o_frame()] * 10, "HO")
        assert k["n_events"] == 0

    def test_alternating_frames_counts_each_run(self):
        # HO in [0], H2O in [1], HO in [2], H2O in [3] → 2 events
        frames = [ho_frame(), h2o_frame(), ho_frame(), h2o_frame()]
        k = reaction_kinetics(frames, "HO")
        assert k["n_events"] == 2


# ============================================================
# lifetime values
# ============================================================

class TestKineticsLifetimes:

    def test_single_event_lifetime_equals_segment_length(self):
        # 4-frame HO block → lifetime = 4
        k = reaction_kinetics([ho_frame()] * 4, "HO")
        assert len(k["lifetimes"]) == 1
        assert k["lifetimes"][0] == pytest.approx(4.0)

    def test_two_events_correct_lifetimes(self):
        # Block 1: 3 frames, block 2: 5 frames
        frames = [ho_frame()] * 3 + [h2o_frame()] * 2 + [ho_frame()] * 5
        k = reaction_kinetics(frames, "HO")
        assert sorted(k["lifetimes"].tolist()) == pytest.approx(sorted([3.0, 5.0]))

    def test_mean_lifetime_correct(self):
        frames = [ho_frame()] * 3 + [h2o_frame()] * 2 + [ho_frame()] * 5
        k = reaction_kinetics(frames, "HO")
        assert k["mean_lifetime"] == pytest.approx(4.0)

    def test_median_lifetime_correct(self):
        frames = [ho_frame()] * 3 + [h2o_frame()] * 2 + [ho_frame()] * 5
        k = reaction_kinetics(frames, "HO")
        assert k["median_lifetime"] == pytest.approx(4.0)

    def test_absent_formula_gives_nan_lifetimes(self):
        k = reaction_kinetics([h2o_frame()] * 10, "HO")
        assert np.isnan(k["mean_lifetime"])
        assert np.isnan(k["median_lifetime"])
        assert len(k["lifetimes"]) == 0

    def test_lifetimes_nonnegative(self):
        frames = [ho_frame()] * 3 + [h2o_frame()] * 2 + [ho_frame()] * 5
        k = reaction_kinetics(frames, "HO")
        assert (k["lifetimes"] >= 0).all()


# ============================================================
# rate
# ============================================================

class TestKineticsRate:

    def test_rate_is_events_per_frame_when_dt_one(self):
        # 2 events in 10 frames → rate = 2/10 = 0.2
        frames = [ho_frame()] * 3 + [h2o_frame()] * 2 + [ho_frame()] * 5
        k = reaction_kinetics(frames, "HO", dt=1.0)
        assert k["rate"] == pytest.approx(2 / 10)

    def test_rate_scales_with_dt(self):
        frames = [ho_frame()] * 3 + [h2o_frame()] * 2 + [ho_frame()] * 5
        k1 = reaction_kinetics(frames, "HO", dt=1.0)
        k2 = reaction_kinetics(frames, "HO", dt=5e-15)
        # rate in s⁻¹ = rate_frames / dt
        assert k2["rate"] == pytest.approx(k1["rate"] / 5e-15)

    def test_lifetimes_scale_with_dt(self):
        frames = [ho_frame()] * 4 + [h2o_frame()] * 2
        k1 = reaction_kinetics(frames, "HO", dt=1.0)
        k2 = reaction_kinetics(frames, "HO", dt=5e-15)
        np.testing.assert_allclose(k2["lifetimes"], k1["lifetimes"] * 5e-15)

    def test_rate_zero_when_formula_absent(self):
        k = reaction_kinetics([h2o_frame()] * 10, "HO")
        assert k["rate"] == pytest.approx(0.0)


# ============================================================
# integration — real 39-water trajectory
# ============================================================

@pytest.fixture(scope="module")
def water_frames():
    from chempiler import ChempilerTrajectory
    t = ChempilerTrajectory(str(TESTS_DIR / "39_water_OH.traj"))
    t.build(cache_file=str(TESTS_DIR / "cache_file.h5"))
    return t.frames


class TestKineticsIntegration:

    def test_ho_has_many_events(self, water_frames):
        k = reaction_kinetics(water_frames, "HO")
        assert k["n_events"] > 100

    def test_ho_mean_lifetime_is_positive(self, water_frames):
        k = reaction_kinetics(water_frames, "HO")
        assert k["mean_lifetime"] > 0

    def test_ho_median_shorter_than_mean(self, water_frames):
        # Lifetime distribution for transient species is right-skewed
        k = reaction_kinetics(water_frames, "HO")
        assert k["median_lifetime"] <= k["mean_lifetime"]

    def test_ho_rate_positive(self, water_frames):
        k = reaction_kinetics(water_frames, "HO", dt=5e-15)
        assert k["rate"] > 0

    def test_h2o_single_event_whole_trajectory(self, water_frames):
        # H2O is present throughout → exactly one lifetime segment
        k = reaction_kinetics(water_frames, "H2O")
        assert k["n_events"] == 1
