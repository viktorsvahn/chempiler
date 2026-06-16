"""Tests for the MSD (mean squared displacement) module.

Unit tests use small synthetic frames with analytically exact expected values:

- A single-atom molecule (formula "O") is used throughout because its COM is
  just the atom position, avoiding any ambiguity about weighted averaging.
- "ballistic" motion (constant velocity v) gives MSD(τ) = v²τ² exactly.
- Stationary molecules give MSD = 0 exactly.

Integration tests load the real 39-water trajectory and check physically
meaningful properties.
"""

import warnings

import numpy as np
import pytest
from pathlib import Path
from ase import Atoms

from chempiler.frame import Frame
from chempiler.msd import msd, _windowed_msd

TESTS_DIR = Path(__file__).parent


# ============================================================
# helpers
# ============================================================

def make_frame(symbols, positions, molecules, cell=(100.0, 100.0, 100.0), pbc=False):
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)
    frame = Frame(atoms=atoms, molecules=molecules)
    frame.build()
    return frame


def stationary_frames(n, pos=(1.0, 2.0, 3.0)):
    """n copies of the same frame: one O atom at a fixed position."""
    return [make_frame(["O"], [list(pos)], [[0]])] * n


def moving_frames(n, v=(1.0, 0.0, 0.0)):
    """n distinct frames: one O atom at t*v for t = 0…n-1 (Å/frame)."""
    v = np.array(v)
    return [make_frame(["O"], [(t * v).tolist()], [[0]]) for t in range(n)]


def two_atom_frames(n, v1=(2.0, 0.0, 0.0), v2=(0.0, 0.0, 0.0)):
    """n frames with two O atoms separated by 50 Å, moving at v1 and v2."""
    v1, v2 = np.array(v1), np.array(v2)
    sep = np.array([50.0, 0.0, 0.0])
    return [
        make_frame(
            ["O", "O"],
            [(t * v1).tolist(), (sep + t * v2).tolist()],
            [[0], [1]],
        )
        for t in range(n)
    ]


def periodic_frames(n, v=1.0, start=8.0, cell_size=10.0):
    """n frames of one O atom at velocity v Å/frame, crossing the x boundary."""
    cell = (cell_size, cell_size, cell_size)
    return [
        make_frame(
            ["O"], [[(start + t * v) % cell_size, 0.0, 0.0]], [[0]],
            cell=cell, pbc=True,
        )
        for t in range(n)
    ]


def h2o_frame(ox=0.0, oy=0.0, oz=0.0):
    """One H2O molecule with O at (ox, oy, oz)."""
    return make_frame(
        ["O", "H", "H"],
        [[ox, oy, oz], [ox + 1.0, oy, oz], [ox, oy + 1.0, oz]],
        [[0, 1, 2]],
    )


def ho_frame():
    return make_frame(["O", "H"], [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]], [[0, 1]])


# ============================================================
# _windowed_msd — direct unit tests
# ============================================================

class TestWindowedMSD:

    def test_stationary_track_gives_zero_msd(self):
        track = np.zeros((10, 3))
        msd_vals, _ = _windowed_msd([track], max_lag=5)
        np.testing.assert_allclose(msd_vals, 0.0, atol=1e-12)

    def test_constant_velocity_ballistic(self):
        # Position r(t) = (t, 0, 0) → MSD(τ) = τ²
        T = 20
        track = np.column_stack([np.arange(T, dtype=float), np.zeros(T), np.zeros(T)])
        msd_vals, _ = _windowed_msd([track], max_lag=5)
        expected = np.arange(1, 6, dtype=float) ** 2
        np.testing.assert_allclose(msd_vals, expected, atol=1e-10)

    def test_nan_for_lags_beyond_track_length(self):
        # Track length 3 → contributions for lag 1 and 2 only
        track = np.zeros((3, 3))
        msd_vals, counts = _windowed_msd([track], max_lag=5)
        assert not np.isnan(msd_vals[0])   # lag 1: T - 1 = 2 samples
        assert not np.isnan(msd_vals[1])   # lag 2: T - 2 = 1 sample
        assert np.isnan(msd_vals[2])       # lag 3: T - 3 = 0 → nan
        assert np.isnan(msd_vals[3])
        assert np.isnan(msd_vals[4])

    def test_two_tracks_averaged(self):
        # Track 1: velocity 2 Å/frame → MSD_1 = 4τ²
        # Track 2: stationary → MSD_2 = 0
        # Average → MSD = 2τ²
        T = 15
        t1 = np.column_stack([2.0 * np.arange(T), np.zeros(T), np.zeros(T)])
        t2 = np.zeros((T, 3))
        msd_vals, _ = _windowed_msd([t1, t2], max_lag=4)
        expected = 2.0 * np.arange(1, 5, dtype=float) ** 2
        np.testing.assert_allclose(msd_vals, expected, atol=1e-10)

    def test_counts_decrease_with_lag(self):
        track = np.zeros((10, 3))
        _, counts = _windowed_msd([track], max_lag=5)
        assert (np.diff(counts) < 0).all()

    def test_empty_track_list_returns_nan(self):
        msd_vals, counts = _windowed_msd([], max_lag=3)
        assert np.all(np.isnan(msd_vals))
        assert np.all(counts == 0)


# ============================================================
# output structure
# ============================================================

class TestMSDOutputStructure:

    def test_returns_three_arrays(self):
        frames = moving_frames(10)
        assert len(msd(frames, "O", max_lag=3)) == 3

    def test_all_arrays_length_equals_max_lag(self):
        frames = moving_frames(10)
        lags, msd_vals, n_samples = msd(frames, "O", max_lag=4)
        assert len(lags) == len(msd_vals) == len(n_samples) == 4

    def test_lags_start_at_one_and_are_consecutive(self):
        frames = moving_frames(10)
        lags, _, _ = msd(frames, "O", max_lag=5)
        np.testing.assert_array_equal(lags, [1, 2, 3, 4, 5])

    def test_output_types_are_numpy(self):
        frames = moving_frames(10)
        lags, msd_vals, n_samples = msd(frames, "O", max_lag=3)
        assert isinstance(lags, np.ndarray)
        assert isinstance(msd_vals, np.ndarray)
        assert isinstance(n_samples, np.ndarray)

    def test_msd_values_non_negative(self):
        frames = moving_frames(10)
        _, msd_vals, _ = msd(frames, "O", max_lag=4)
        valid = msd_vals[~np.isnan(msd_vals)]
        assert (valid >= 0).all()


# ============================================================
# stationary molecule — MSD must be exactly zero
# ============================================================

class TestMSDStationary:

    def test_single_stationary_molecule_msd_zero(self):
        frames = stationary_frames(10)
        _, msd_vals, _ = msd(frames, "O", max_lag=4)
        np.testing.assert_allclose(msd_vals, 0.0, atol=1e-12)

    def test_stationary_at_nonzero_position_msd_zero(self):
        frames = stationary_frames(10, pos=(5.3, 2.1, 9.8))
        _, msd_vals, _ = msd(frames, "O", max_lag=3)
        np.testing.assert_allclose(msd_vals, 0.0, atol=1e-12)


# ============================================================
# ballistic (constant velocity) motion — MSD = v²τ²
# ============================================================

class TestMSDBallistic:

    def test_unit_velocity_along_x(self):
        # v = (1, 0, 0) Å/frame → MSD(τ) = τ²
        frames = moving_frames(20, v=(1.0, 0.0, 0.0))
        lags, msd_vals, _ = msd(frames, "O", max_lag=5)
        np.testing.assert_allclose(msd_vals, lags.astype(float) ** 2, atol=1e-10)

    def test_faster_velocity_gives_larger_msd(self):
        _, msd_slow, _ = msd(moving_frames(20, v=(1.0, 0.0, 0.0)), "O", max_lag=4)
        _, msd_fast, _ = msd(moving_frames(20, v=(2.0, 0.0, 0.0)), "O", max_lag=4)
        assert (msd_fast > msd_slow).all()

    def test_velocity_in_all_three_dimensions(self):
        # v = (1, 1, 1) → |v|² = 3 → MSD(τ) = 3τ²
        frames = moving_frames(20, v=(1.0, 1.0, 1.0))
        lags, msd_vals, _ = msd(frames, "O", max_lag=5)
        np.testing.assert_allclose(msd_vals, 3.0 * lags.astype(float) ** 2, atol=1e-10)

    def test_two_molecules_averaged(self):
        # Atom 0 at v=2 Å/frame, atom 1 stationary → MSD = (4τ² + 0) / 2 = 2τ²
        frames = two_atom_frames(20, v1=(2.0, 0.0, 0.0), v2=(0.0, 0.0, 0.0))
        lags, msd_vals, _ = msd(frames, "O", max_lag=4)
        np.testing.assert_allclose(msd_vals, 2.0 * lags.astype(float) ** 2, atol=1e-10)


# ============================================================
# max_lag parameter
# ============================================================

class TestMSDMaxLag:

    def test_explicit_max_lag_sets_output_length(self):
        frames = moving_frames(20)
        lags, _, _ = msd(frames, "O", max_lag=7)
        assert len(lags) == 7

    def test_default_max_lag_is_half_longest_segment(self):
        # 20-frame trajectory → one segment of 20 → default max_lag = 10
        frames = moving_frames(20)
        lags, _, _ = msd(frames, "O")
        assert len(lags) == 10

    def test_lags_beyond_track_length_are_nan(self):
        # Track length 5: valid for lag 1–4, nan from lag 5 onwards
        frames = moving_frames(5)
        _, msd_vals, _ = msd(frames, "O", max_lag=8)
        assert not np.isnan(msd_vals[0])  # lag 1 (T-1=4 samples)
        assert not np.isnan(msd_vals[3])  # lag 4 (T-4=1 sample)
        assert np.isnan(msd_vals[4])      # lag 5 (T-5=0)
        assert np.isnan(msd_vals[7])      # lag 8


# ============================================================
# lifetime segment handling
# ============================================================

class TestMSDLifetimeSegments:

    def test_absent_formula_raises(self):
        frames = stationary_frames(10)
        with pytest.raises(ValueError, match="not found"):
            msd(frames, "X")

    def test_all_single_frame_segments_raises(self):
        # H2O alternates with HO → H2O segments of length 1 → no tracks buildable
        frames = [h2o_frame(), ho_frame()] * 5
        with pytest.raises(ValueError, match="No molecule tracks"):
            msd(frames, "H2O")

    def test_gaps_between_segments_respected(self):
        # H2O in frames 0-4 and 8-12 (absent in 5-7); H2O is stationary → MSD = 0
        frames = [h2o_frame()] * 5 + [ho_frame()] * 3 + [h2o_frame()] * 5
        _, msd_vals, _ = msd(frames, "H2O", max_lag=2)
        np.testing.assert_allclose(msd_vals, 0.0, atol=1e-12)

    def test_n_samples_decreases_with_lag(self):
        frames = moving_frames(20)
        _, _, n = msd(frames, "O", max_lag=5)
        assert (np.diff(n) <= 0).all()

    def test_n_samples_positive_for_small_lags(self):
        frames = moving_frames(10)
        _, _, n = msd(frames, "O", max_lag=3)
        assert (n > 0).all()

    def test_molecule_appearing_mid_segment_produces_shorter_track(self):
        # Segment where a second O atom appears at frame 2 — should not crash
        # and should produce a non-empty result
        f0 = make_frame(["O"], [[0.0, 0.0, 0.0]], [[0]])
        f1 = make_frame(["O"], [[1.0, 0.0, 0.0]], [[0]])
        f2 = make_frame(["O", "O"], [[2.0, 0.0, 0.0], [50.0, 0.0, 0.0]], [[0], [1]])
        f3 = make_frame(["O", "O"], [[3.0, 0.0, 0.0], [50.0, 0.0, 0.0]], [[0], [1]])
        lags, msd_vals, _ = msd([f0, f1, f2, f3], "O", max_lag=2)
        assert len(lags) == 2
        assert not np.isnan(msd_vals[0])


# ============================================================
# PBC / minimum image convention
# ============================================================

class TestMSDPBC:

    def test_boundary_crossing_gives_correct_msd(self):
        # O moves at 1 Å/frame, crosses x=10 boundary between frames 1 and 2.
        # Unwrapped displacement must give MSD(τ) = τ², not (-cell + τ)².
        frames = periodic_frames(n=12, v=1.0, start=8.0, cell_size=10.0)
        lags, msd_vals, _ = msd(frames, "O", max_lag=4)
        np.testing.assert_allclose(msd_vals, lags.astype(float) ** 2, atol=1e-10)

    def test_large_cell_pbc_matches_nopbc(self):
        # With a cell much larger than the displacement, PBC and no-PBC must agree
        frames_pbc   = periodic_frames(n=5, v=1.0, start=0.0, cell_size=1000.0)
        frames_nopbc = moving_frames(5, v=(1.0, 0.0, 0.0))
        _, msd_pbc,   _ = msd(frames_pbc,   "O", max_lag=3)
        _, msd_nopbc, _ = msd(frames_nopbc, "O", max_lag=3)
        np.testing.assert_allclose(msd_pbc, msd_nopbc, atol=1e-8)


# ============================================================
# correlation_time warning
# ============================================================

class TestMSDCorrelationTimeWarning:

    def _user_warnings(self, frames, formula, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            msd(frames, formula, **kwargs)
        return [w for w in caught if issubclass(w.category, UserWarning)]

    def test_no_warning_when_correlation_time_not_set(self):
        assert len(self._user_warnings(stationary_frames(3), "O", max_lag=1)) == 0

    def test_warning_fires_for_short_segments(self):
        # Segment length 2, threshold = 1 × 5 = 5 → short → warn
        ws = self._user_warnings(stationary_frames(2), "O", max_lag=1,
                                 correlation_time=1)
        assert len(ws) == 1
        assert issubclass(ws[0].category, UserWarning)

    def test_no_warning_for_long_segments(self):
        # Segment length 20, threshold = 1 × 5 = 5 → 20 ≥ 5 → no warn
        ws = self._user_warnings(stationary_frames(20), "O", max_lag=5,
                                 correlation_time=1)
        assert len(ws) == 0

    def test_warning_at_threshold_boundary(self):
        # Segment length exactly equals threshold → condition is <, not ≤ → no warn
        # threshold = 2 × 5 = 10, segment length = 10
        ws = self._user_warnings(stationary_frames(10), "O", max_lag=4,
                                 correlation_time=2, buffer=5)
        assert len(ws) == 0

    def test_warning_below_threshold_boundary(self):
        # Segment length 9 < threshold 10 → warn
        ws = self._user_warnings(stationary_frames(9), "O", max_lag=4,
                                 correlation_time=2, buffer=5)
        assert len(ws) == 1

    def test_buffer_scales_threshold(self):
        # Segment 6, correlation_time=1: buffer=5 → thresh=5 → no warn;
        #                                buffer=10 → thresh=10 → warn
        no_warn = self._user_warnings(stationary_frames(6), "O", max_lag=2,
                                      correlation_time=1, buffer=5)
        warn    = self._user_warnings(stationary_frames(6), "O", max_lag=2,
                                      correlation_time=1, buffer=10)
        assert len(no_warn) == 0
        assert len(warn) == 1

    def test_warning_message_contains_formula_and_counts(self):
        ws = self._user_warnings(stationary_frames(2), "O", max_lag=1,
                                 correlation_time=1)
        msg = str(ws[0].message)
        assert "'O'" in msg
        assert "1 of 1" in msg  # 1 short segment out of 1 total


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


class TestMSDIntegration:

    def test_h2o_msd_is_positive(self, water_frames):
        _, msd_vals, _ = msd(water_frames, "H2O", max_lag=10)
        assert (msd_vals > 0).all()

    def test_h2o_msd_increases_with_lag(self, water_frames):
        _, msd_vals, _ = msd(water_frames, "H2O", max_lag=20)
        assert msd_vals[-1] > msd_vals[0]

    def test_ho_msd_returns_valid_values(self, water_frames):
        lags, msd_vals, n = msd(water_frames, "HO", max_lag=5)
        assert len(lags) == 5
        assert (msd_vals[~np.isnan(msd_vals)] >= 0).all()
        assert n[0] > 0

    def test_ho_warning_fires_with_correlation_time(self, water_frames):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            msd(water_frames, "HO", max_lag=3, correlation_time=10)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1
