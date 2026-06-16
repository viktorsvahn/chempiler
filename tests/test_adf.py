"""Tests for chempiler.adf.

Synthetic frames with known angles are used to verify the peak position and
normalization. Integration tests check physically meaningful properties of the
H-O-H angle distribution in the real trajectory.
"""

import numpy as np
import pytest
from pathlib import Path
from ase import Atoms

from chempiler.frame import Frame
from chempiler.adf import adf

TESTS_DIR = Path(__file__).parent
LARGE_CELL = (50.0, 50.0, 50.0)


# ============================================================
# helpers
# ============================================================

def make_frame(symbols, positions, molecules, cell=LARGE_CELL, pbc=True):
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)
    frame = Frame(atoms=atoms, molecules=molecules)
    frame.build()
    return frame


def right_angle_h2o():
    """H2O with H-O-H = 90° exactly.

    O at origin, H1 along x, H2 along y → cos(angle) = 0 → angle = 90°.
    """
    return make_frame(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        molecules=[[0, 1, 2]],
    )


def linear_h2o():
    """Artificial H-O-H with angle = 180°: H-O-H collinear."""
    return make_frame(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        molecules=[[0, 1, 2]],
    )


def single_h_frame():
    """Molecule with only one H bonded to O — no angle possible."""
    return make_frame(
        symbols=["O", "H"],
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        molecules=[[0, 1]],
    )


# ============================================================
# output structure
# ============================================================

class TestADFOutputStructure:

    def test_returns_two_arrays(self):
        result = adf([right_angle_h2o()] * 5, center="O", neighbors="H")
        assert len(result) == 2

    def test_arrays_are_numpy(self):
        angles, density = adf([right_angle_h2o()] * 5, center="O", neighbors="H")
        assert isinstance(angles, np.ndarray)
        assert isinstance(density, np.ndarray)

    def test_arrays_same_length(self):
        angles, density = adf([right_angle_h2o()] * 5, center="O", neighbors="H")
        assert len(angles) == len(density)

    def test_length_equals_bins(self):
        angles, density = adf([right_angle_h2o()] * 5, center="O", neighbors="H",
                               bins=60)
        assert len(angles) == 60

    def test_bin_centers_within_angle_range(self):
        lo, hi = 80.0, 130.0
        angles, _ = adf([right_angle_h2o()] * 5, center="O", neighbors="H",
                        angle_range=(lo, hi))
        assert angles[0] >= lo
        assert angles[-1] <= hi


# ============================================================
# known angles
# ============================================================

class TestADFKnownAngles:

    def test_peak_at_90_degrees_for_right_angle_h2o(self):
        angles, density = adf([right_angle_h2o()] * 50,
                               center="O", neighbors="H",
                               bins=100, angle_range=(80, 100))
        peak_angle = angles[density.argmax()]
        assert abs(peak_angle - 90.0) < 2.0, f"Peak at {peak_angle}°, expected 90°"

    def test_peak_at_180_for_linear_molecule(self):
        angles, density = adf([linear_h2o()] * 50,
                               center="O", neighbors="H",
                               bins=50, angle_range=(170, 180))
        # All angles should fall in the 170-180 range
        assert density.sum() > 0

    def test_only_one_molecule_all_angles_same(self):
        # All frames identical → single sharp peak
        frames = [right_angle_h2o()] * 30
        angles, density = adf(frames, center="O", neighbors="H",
                               bins=100, angle_range=(80, 100))
        # One bin should have all the counts
        assert (density > 0).sum() <= 2   # at most 2 adjacent bins (bin-edge effects)


# ============================================================
# formula selector
# ============================================================

class TestADFFormula:

    def test_formula_none_and_formula_h2o_agree_when_only_h2o(self):
        frames = [right_angle_h2o()] * 10
        a1, d1 = adf(frames, "O", "H", formula=None)
        a2, d2 = adf(frames, "O", "H", formula="H2O")
        np.testing.assert_allclose(d1, d2)

    def test_formula_filter_excludes_other_molecules(self):
        # Frame with one H2O (90°) and one HO (no angle possible)
        frame_mixed = make_frame(
            symbols=["O", "H", "H", "O", "H"],
            positions=[
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                [20.0, 0.0, 0.0], [21.0, 0.0, 0.0],
            ],
            molecules=[[0, 1, 2], [3, 4]],
        )
        # HO has only one H → no angle; result should be same as H2O-only
        frames_mixed = [frame_mixed] * 10
        frames_pure  = [right_angle_h2o()] * 10

        a_m, d_m = adf(frames_mixed, "O", "H", formula="H2O", bins=50, angle_range=(80, 100))
        a_p, d_p = adf(frames_pure,  "O", "H", formula="H2O", bins=50, angle_range=(80, 100))
        np.testing.assert_allclose(d_m, d_p)

    def test_formula_absent_returns_empty_arrays(self):
        angles, density = adf([right_angle_h2o()] * 5, "O", "H", formula="HO")
        assert len(angles) == 0
        assert len(density) == 0


# ============================================================
# edge cases
# ============================================================

class TestADFEdgeCases:

    def test_single_h_no_angle_returns_empty(self):
        angles, density = adf([single_h_frame()] * 5, center="O", neighbors="H")
        assert len(angles) == 0 or density.sum() == 0

    def test_no_center_element_returns_empty(self):
        frames = [right_angle_h2o()] * 5
        angles, density = adf(frames, center="N", neighbors="H")
        assert len(angles) == 0

    def test_single_frame_does_not_crash(self):
        angles, density = adf([right_angle_h2o()], center="O", neighbors="H")
        assert len(angles) == len(density)


# ============================================================
# normalisation
# ============================================================

class TestADFNormalisation:

    def test_density_nonnegative(self):
        _, density = adf([right_angle_h2o()] * 20, center="O", neighbors="H")
        assert (density >= 0).all()

    def test_density_integrates_to_one(self):
        lo, hi = 80.0, 100.0
        bins = 80
        angles, density = adf([right_angle_h2o()] * 20, center="O", neighbors="H",
                               bins=bins, angle_range=(lo, hi))
        da = (hi - lo) / bins
        integral = (density * da).sum()
        assert abs(integral - 1.0) < 0.05   # allow 5% tolerance


# ============================================================
# integration — real 39-water trajectory
# ============================================================

@pytest.fixture(scope="module")
def water_frames():
    from chempiler import ChempilerTrajectory
    t = ChempilerTrajectory(str(TESTS_DIR / "39_water_OH.traj"))
    t.build(cache_file=str(TESTS_DIR / "cache_file.h5"))
    return t.frames


class TestADFIntegration:

    def test_h2o_peak_near_104_degrees(self, water_frames):
        angles, density = adf(water_frames, center="O", neighbors="H",
                               formula="H2O", bins=100, angle_range=(90, 120))
        peak = angles[density.argmax()]
        assert 100.0 < peak < 115.0, f"H-O-H peak at {peak:.1f}°, expected ~104°"

    def test_density_nonneg_real_traj(self, water_frames):
        _, density = adf(water_frames, "O", "H", formula="H2O")
        assert (density >= 0).all()

    def test_density_integrates_to_one_real_traj(self, water_frames):
        lo, hi = 80.0, 140.0
        bins = 90
        _, density = adf(water_frames, "O", "H", formula="H2O",
                         bins=bins, angle_range=(lo, hi))
        da = (hi - lo) / bins
        integral = (density * da).sum()
        assert abs(integral - 1.0) < 0.05
