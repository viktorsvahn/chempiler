"""Tests for chempiler.vanhove.van_hove.

Synthetic frames use single H2O molecules at controlled positions so that
displacement magnitudes at each lag are known exactly. The key checks are:
- Output shape is (len(lags), len(r))
- For a stationary molecule the peak is in the first bin (r ≈ 0)
- For a moving molecule the peak shifts outward with increasing lag
- Normalisation: ∫ G_s 4π r² dr ≈ 1

Integration tests use the real trajectory for sanity checks.
"""

import numpy as np
import pytest
from pathlib import Path
from ase import Atoms

from chempiler.frame import Frame
from chempiler.vanhove import van_hove

TESTS_DIR = Path(__file__).parent


# ============================================================
# helpers
# ============================================================

def make_frame(symbols, positions, molecules, cell=(100.0, 100.0, 100.0), pbc=False):
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)
    frame = Frame(atoms=atoms, molecules=molecules)
    frame.build()
    return frame


def h2o_frame(ox=0.0, oy=0.0, oz=0.0):
    """One H2O molecule with O at (ox, oy, oz)."""
    return make_frame(
        ["O", "H", "H"],
        [[ox, oy, oz], [ox + 1.0, oy, oz], [ox, oy + 1.0, oz]],
        [[0, 1, 2]],
    )


def ho_frame():
    return make_frame(
        ["O", "H"],
        [[50.0, 0.0, 0.0], [51.0, 0.0, 0.0]],
        [[0, 1]],
    )


def moving_h2o_frames(n, v=0.5):
    """n frames with one H2O moving at v Å/frame along x."""
    return [h2o_frame(ox=t * v) for t in range(n)]


def stationary_h2o_frames(n):
    return [h2o_frame(ox=0.0)] * n


# ============================================================
# output structure
# ============================================================

class TestVanHoveOutputStructure:

    def test_returns_two_items(self):
        result = van_hove(moving_h2o_frames(20), "H2O", lags=[1, 5])
        assert len(result) == 2

    def test_r_is_ndarray(self):
        r, G = van_hove(moving_h2o_frames(20), "H2O", lags=[1])
        assert isinstance(r, np.ndarray)

    def test_G_is_ndarray(self):
        r, G = van_hove(moving_h2o_frames(20), "H2O", lags=[1])
        assert isinstance(G, np.ndarray)

    def test_G_shape_lags_by_r(self):
        lags = [1, 5, 10]
        r, G = van_hove(moving_h2o_frames(30), "H2O", lags=lags, rmax=5.0, dr=0.1)
        assert G.shape == (len(lags), len(r))

    def test_r_bin_centers_start_at_dr_over_two(self):
        dr = 0.2
        r, _ = van_hove(moving_h2o_frames(20), "H2O", lags=[1], rmax=4.0, dr=dr)
        assert abs(r[0] - dr / 2) < 1e-10

    def test_r_spacing_equals_dr(self):
        dr = 0.15
        r, _ = van_hove(moving_h2o_frames(20), "H2O", lags=[1], rmax=3.0, dr=dr)
        np.testing.assert_allclose(np.diff(r), dr, atol=1e-10)

    def test_r_max_within_rmax(self):
        rmax = 4.0
        r, _ = van_hove(moving_h2o_frames(20), "H2O", lags=[1], rmax=rmax, dr=0.1)
        assert r[-1] < rmax + 0.1


# ============================================================
# physical behaviour
# ============================================================

class TestVanHovePhysics:

    def test_G_nonnegative(self):
        r, G = van_hove(moving_h2o_frames(30), "H2O", lags=[1, 5, 15],
                        rmax=6.0, dr=0.1)
        assert (G >= 0).all()

    def test_stationary_peak_in_first_bin(self):
        # Zero displacement → all mass in r ≈ 0 bin
        r, G = van_hove(stationary_h2o_frames(20), "H2O", lags=[1],
                        rmax=3.0, dr=0.1)
        assert G[0, 0] == G[0].max()

    def test_peak_shifts_outward_with_lag(self):
        # H2O at v=0.5 Å/frame: lag=1 → displacement ≈ 0.5; lag=10 → ≈ 5
        frames = moving_h2o_frames(50, v=0.5)
        r, G = van_hove(frames, "H2O", lags=[1, 10], rmax=8.0, dr=0.1)
        peak_1  = r[G[0].argmax()]
        peak_10 = r[G[1].argmax()]
        assert peak_10 > peak_1

    def test_peak_near_expected_displacement(self):
        # H2O at v=1.0 Å/frame; lag=1 → displacement = 1.0
        frames = moving_h2o_frames(30, v=1.0)
        r, G = van_hove(frames, "H2O", lags=[1], rmax=5.0, dr=0.1)
        peak = r[G[0].argmax()]
        assert abs(peak - 1.0) < 0.2, f"Peak at {peak:.2f} Å, expected ~1.0 Å"

    def test_normalisation_approximately_one(self):
        # ∫ G_s(r) 4π r² dr ≈ 1
        frames = moving_h2o_frames(50, v=0.3)
        dr = 0.05
        r, G = van_hove(frames, "H2O", lags=[1], rmax=6.0, dr=dr)
        shell = 4 * np.pi * r ** 2 * dr
        integral = (G[0] * shell).sum()
        assert abs(integral - 1.0) < 0.1


# ============================================================
# error handling
# ============================================================

class TestVanHoveErrors:

    def test_missing_formula_raises_value_error(self):
        with pytest.raises(ValueError, match="not found"):
            van_hove(stationary_h2o_frames(10), "HO", lags=[1])

    def test_lag_beyond_track_length_gives_zero_row(self):
        # Track length = 5; lag = 10 → no samples → G row should be all zeros
        frames = moving_h2o_frames(5, v=0.5)
        r, G = van_hove(frames, "H2O", lags=[10], rmax=5.0, dr=0.1)
        assert G[0].sum() == 0.0


# ============================================================
# integration — real 39-water trajectory
# ============================================================

@pytest.fixture(scope="module")
def water_frames():
    from chempiler import ChempilerTrajectory
    t = ChempilerTrajectory(str(TESTS_DIR / "39_water_OH.traj"))
    t.build(cache_file=str(TESTS_DIR / "cache_file.h5"))
    return t.frames


class TestVanHoveIntegration:

    def test_shape_correct_real_trajectory(self, water_frames):
        lags = [1, 10, 50]
        r, G = van_hove(water_frames, "H2O", lags=lags, rmax=6.0, dr=0.1)
        assert G.shape == (len(lags), len(r))

    def test_G_nonneg_real_trajectory(self, water_frames):
        _, G = van_hove(water_frames, "H2O", lags=[1, 20], rmax=5.0, dr=0.1)
        assert (G >= 0).all()

    def test_mean_displacement_grows_with_lag(self, water_frames):
        # G_s(r) is monotonically decreasing for Gaussian displacements, so
        # comparing peak positions is meaningless. Use the mean displacement
        # <r> = ∫ r G_s 4πr² dr / ∫ G_s 4πr² dr instead.
        dr = 0.1
        r, G = van_hove(water_frames, "H2O", lags=[1, 200], rmax=8.0, dr=dr)
        shell = 4 * np.pi * r ** 2 * dr

        def mean_r(Gi):
            total = (Gi * shell).sum()
            return (r * Gi * shell).sum() / total if total > 0 else 0.0

        assert mean_r(G[1]) > mean_r(G[0])

    def test_normalisation_real_trajectory(self, water_frames):
        dr = 0.1
        r, G = van_hove(water_frames, "H2O", lags=[10], rmax=6.0, dr=dr)
        shell = 4 * np.pi * r ** 2 * dr
        integral = (G[0] * shell).sum()
        assert 0.5 < integral < 1.5
