"""Tests for chempiler.tetrahedral.tetrahedral_order.

The key unit test uses a perfect tetrahedral arrangement of O atoms where the
expected value q = 1.0 is exact. Integration tests check physically meaningful
ranges against the real trajectory.
"""

import numpy as np
import pytest
from pathlib import Path
from ase import Atoms

from chempiler.frame import Frame
from chempiler.tetrahedral import tetrahedral_order

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


def perfect_tet_frame(d=2.0):
    """Five O atoms: one centre + four in tetrahedral positions at distance d.

    The four corner directions are mutually at 109.47°, giving q = 1 for the
    centre atom. The corner-to-corner distance is d × sqrt(8/3) ≈ 1.633d,
    so with rcut < 1.633d each corner has only 1 neighbour (the centre) and
    is excluded from the mean (< n_neighbours = 4).
    """
    s3 = np.sqrt(3.0)
    dirs = np.array([
        [ 1,  1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
    ], dtype=float) / s3

    positions = [[0.0, 0.0, 0.0]] + (d * dirs).tolist()
    mols = [[i] for i in range(5)]
    return make_frame(["O"] * 5, positions, mols)


def random_o_frame(seed=42):
    """Eight O atoms at random positions — expect q well below 1."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, 5, size=(8, 3)).tolist()
    mols = [[i] for i in range(8)]
    return make_frame(["O"] * 8, positions, mols)


# ============================================================
# output structure
# ============================================================

class TestTetrahedralOutputStructure:

    def test_returns_ndarray(self):
        result = tetrahedral_order([perfect_tet_frame()] * 3)
        assert isinstance(result, np.ndarray)

    def test_length_equals_n_frames(self):
        n = 5
        result = tetrahedral_order([perfect_tet_frame()] * n)
        assert len(result) == n

    def test_values_float(self):
        result = tetrahedral_order([perfect_tet_frame()] * 2)
        assert result.dtype.kind == "f"


# ============================================================
# known values
# ============================================================

class TestTetrahedralKnownValues:

    def test_perfect_tetrahedron_gives_q_one(self):
        # rcut=2.5 captures centre-corner (d=2.0) but not corner-corner (≈3.27)
        q = tetrahedral_order([perfect_tet_frame(d=2.0)], rcut=2.5)
        # Only the centre atom qualifies (4 neighbours); corners have <4 → excluded
        assert q[0] == pytest.approx(1.0, abs=1e-6)

    def test_random_positions_q_less_than_one(self):
        q = tetrahedral_order([random_o_frame()] * 3, rcut=6.0)
        assert np.nanmean(q) < 1.0

    def test_q_in_valid_range(self):
        # Theoretical bounds: q ∈ [-3, 1]
        # Maximum sum = 6 × (1 + 1/3)² = 32/3 → minimum q = 1 − (3/8)(32/3) = −3
        q = tetrahedral_order([random_o_frame()] * 5, rcut=5.0)
        valid = q[~np.isnan(q)]
        assert (valid >= -3.0 - 1e-9).all()
        assert (valid <= 1.0 + 1e-9).all()

    def test_consistent_across_identical_frames(self):
        frames = [perfect_tet_frame()] * 5
        q = tetrahedral_order(frames, rcut=2.5)
        np.testing.assert_allclose(q, q[0], atol=1e-10)


# ============================================================
# rcut and n_neighbours behaviour
# ============================================================

class TestTetrahedralParameters:

    def test_too_tight_rcut_gives_nan(self):
        # rcut < d (2.0) → no atom has 4 neighbours → all nan
        q = tetrahedral_order([perfect_tet_frame(d=2.0)], rcut=1.0)
        assert np.all(np.isnan(q))

    def test_fewer_than_n_neighbours_atoms_gives_nan(self):
        # Only 3 O atoms total; need 4+1=5 for n_neighbours=4 → nan
        frame = make_frame(
            ["O"] * 3,
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0], [1], [2]],
        )
        q = tetrahedral_order([frame])
        assert np.all(np.isnan(q))

    def test_element_filter_respected(self):
        # Frame with O and H; computing for element="H" should use H-H neighbours
        frame = make_frame(
            ["O", "H", "H", "H", "H", "H"],
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            [[i] for i in range(6)],
        )
        # Should not raise; may return nan if geometry doesn't satisfy rcut
        q_h = tetrahedral_order([frame], element="H", rcut=3.0)
        assert len(q_h) == 1


# ============================================================
# integration — real 39-water trajectory
# ============================================================

@pytest.fixture(scope="module")
def water_frames():
    from chempiler import ChempilerTrajectory
    t = ChempilerTrajectory(str(TESTS_DIR / "39_water_OH.traj"))
    t.build(cache_file=str(TESTS_DIR / "cache_file.h5"))
    return t.frames


class TestTetrahedralIntegration:

    def test_mean_q_between_zero_and_one(self, water_frames):
        q = tetrahedral_order(water_frames[:50])
        mean_q = np.nanmean(q)
        assert 0.0 < mean_q < 1.0

    def test_mean_q_reasonable_for_hot_water(self, water_frames):
        # Reactive (high-T) water: expect q lower than ambient liquid (0.57)
        q = tetrahedral_order(water_frames[:50])
        assert np.nanmean(q) < 0.8

    def test_no_nan_with_default_rcut(self, water_frames):
        # All frames have 39 O atoms → plenty of neighbours → no nan expected
        q = tetrahedral_order(water_frames[:20])
        assert not np.any(np.isnan(q))

    def test_output_length_matches_input(self, water_frames):
        subset = water_frames[:30]
        q = tetrahedral_order(subset)
        assert len(q) == len(subset)
