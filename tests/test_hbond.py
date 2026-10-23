"""Tests for chempiler.hbond: hbond_count and hbond_acf.

Synthetic frames are constructed so that H-bond counts are analytically known.
H-bond criterion: H···O_acceptor distance < r_HA, where H is covalently bonded
to a donor O (both in the same molecule).
"""

import numpy as np
import pytest
from pathlib import Path
from ase import Atoms

from chempiler.frame import Frame
from chempiler.hbond import hbond_count, hbond_acf, hbonds_at

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


def two_hbond_frame(r_HA_target=1.2):
    """H2O with H1 and H2 each donating one H-bond to a nearby acceptor O.

    H1 (at 1.0, 0, 0) → acceptor O2 at (1.0 + r_HA_target, 0, 0)
    H2 (at 0, 1.0, 0) → acceptor O3 at (0, 1.0 + r_HA_target, 0)

    r_HA_target is chosen so both H-bonds are within cutoff (default 2.4 Å)
    but the cross pairs (H1→O3 and H2→O2) are just beyond it.
    """
    a = 1.0 + r_HA_target
    return make_frame(
        symbols=["O", "H", "H", "O", "O"],
        positions=[
            [0.0, 0.0, 0.0],  # O1 donor
            [1.0, 0.0, 0.0],  # H1
            [0.0, 1.0, 0.0],  # H2
            [a,   0.0, 0.0],  # O2 acceptor for H1
            [0.0, a,   0.0],  # O3 acceptor for H2
        ],
        molecules=[[0, 1, 2], [3], [4]],
    )


def zero_hbond_frame():
    """H2O with an acceptor O far away — no H-bonds."""
    return make_frame(
        symbols=["O", "H", "H", "O"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [20.0, 0.0, 0.0],   # too far
        ],
        molecules=[[0, 1, 2], [3]],
    )


def no_acceptor_frame():
    """Isolated H2O with no other O — no H-bonds possible."""
    return make_frame(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        molecules=[[0, 1, 2]],
    )


# ============================================================
# hbond_count — output structure
# ============================================================

class TestHbondCountOutputStructure:

    def test_returns_ndarray(self):
        result = hbond_count([two_hbond_frame()] * 3)
        assert isinstance(result, np.ndarray)

    def test_length_equals_n_frames(self):
        n = 7
        result = hbond_count([two_hbond_frame()] * n)
        assert len(result) == n

    def test_values_are_non_negative_ints(self):
        result = hbond_count([two_hbond_frame(), zero_hbond_frame()])
        assert (result >= 0).all()


# ============================================================
# hbond_count — known values
# ============================================================

class TestHbondCountValues:

    def test_two_hbonds_detected(self):
        # r_HA_target=1.2 → each H is 1.2 Å from its acceptor O → 2 H-bonds
        count = hbond_count([two_hbond_frame(r_HA_target=1.2)], r_HA=2.4)
        assert count[0] == 2

    def test_zero_hbonds_when_acceptor_far(self):
        count = hbond_count([zero_hbond_frame()], r_HA=2.4)
        assert count[0] == 0

    def test_zero_hbonds_with_no_acceptor(self):
        count = hbond_count([no_acceptor_frame()], r_HA=2.4)
        assert count[0] == 0

    def test_tighter_cutoff_removes_bonds(self):
        # H···O distance = 1.2 Å; cutoff 1.0 should exclude it
        count = hbond_count([two_hbond_frame(r_HA_target=1.2)], r_HA=1.0)
        assert count[0] == 0

    def test_consistent_across_identical_frames(self):
        n = 10
        counts = hbond_count([two_hbond_frame()] * n)
        assert (counts == counts[0]).all()


# ============================================================
# hbond_acf — output structure
# ============================================================

class TestHbondACFOutputStructure:

    def test_returns_two_arrays(self):
        result = hbond_acf([two_hbond_frame()] * 10, max_lag=5)
        assert len(result) == 2

    def test_lags_length_equals_max_lag(self):
        lags, C = hbond_acf([two_hbond_frame()] * 10, max_lag=5)
        assert len(lags) == 5

    def test_c_length_equals_max_lag(self):
        lags, C = hbond_acf([two_hbond_frame()] * 10, max_lag=5)
        assert len(C) == 5

    def test_lags_start_at_one(self):
        lags, _ = hbond_acf([two_hbond_frame()] * 10, max_lag=5)
        assert lags[0] == 1

    def test_lags_are_consecutive(self):
        lags, _ = hbond_acf([two_hbond_frame()] * 10, max_lag=5)
        np.testing.assert_array_equal(lags, np.arange(1, 6))

    def test_c_values_in_zero_one(self):
        _, C = hbond_acf([two_hbond_frame()] * 20, max_lag=10)
        valid = C[~np.isnan(C)]
        assert (valid >= -1e-10).all()
        assert (valid <= 1.0 + 1e-10).all()


# ============================================================
# hbond_acf — physical behaviour
# ============================================================

class TestHbondACFBehaviour:

    def test_identical_frames_give_acf_one(self):
        # Same H-bonds in every frame → C(τ) = 1 for all τ
        _, C = hbond_acf([two_hbond_frame()] * 20, max_lag=10)
        np.testing.assert_allclose(C, 1.0, atol=1e-10)

    def test_no_bonds_returns_zeros(self):
        _, C = hbond_acf([zero_hbond_frame()] * 10, max_lag=5)
        # No bonds → C = 0 (not nan, because we skip frames with no bonds)
        assert not np.any(np.isnan(C))

    def test_acf_real_trajectory_decays(self):
        # On the real trajectory the ACF should start high and decrease
        from chempiler import ChempilerTrajectory
        traj = ChempilerTrajectory(str(TESTS_DIR / "39_water_OH.traj"))
        traj.build(cache_file=str(TESTS_DIR / "cache_file.h5"))
        lags, C = hbond_acf(traj.frames[:200], max_lag=50)
        assert C[0] > C[-1], "ACF should decay over time"


# ============================================================
# integration — real 39-water trajectory
# ============================================================

@pytest.fixture(scope="module")
def water_frames():
    from chempiler import ChempilerTrajectory
    t = ChempilerTrajectory(str(TESTS_DIR / "39_water_OH.traj"))
    t.build(cache_file=str(TESTS_DIR / "cache_file.h5"))
    return t.frames


class TestHbondIntegration:

    def test_mean_hbond_count_positive(self, water_frames):
        counts = hbond_count(water_frames[:100])
        assert counts.mean() > 0

    def test_mean_hbond_count_reasonable(self, water_frames):
        # 39 water molecules: expect roughly 39–80 H-bonds (1–2 per molecule)
        counts = hbond_count(water_frames[:100])
        assert 20 < counts.mean() < 120

    def test_acf_starts_near_one(self, water_frames):
        _, C = hbond_acf(water_frames[:300], max_lag=50)
        assert C[0] > 0.5

    def test_acf_arrays_correct_shape(self, water_frames):
        max_lag = 30
        lags, C = hbond_acf(water_frames[:200], max_lag=max_lag)
        assert len(lags) == max_lag
        assert len(C) == max_lag


# ============================================================
# hbonds_at — H-bonds of a chosen atom or molecule
# ============================================================

def donor_water(acceptor_symbol="O", acceptor_pos=(2.86, 0.0, 0.0), acceptor_h=True):
    """Water 0 (O0, H1, H2) with H1 along +x; an acceptor molecule at acceptor_pos.

    H1...acceptor is 1.9 Å and linear for the default position. The
    acceptor's own H atoms (if any) point away, so they bond to nothing.
    """
    ax, ay, az = acceptor_pos
    symbols = ["O", "H", "H", acceptor_symbol]
    positions = [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0], [ax, ay, az]]
    molecules = [[0, 1, 2], [3]]
    if acceptor_h:
        symbols += ["H", "H"]
        positions += [[ax + 0.96, ay, az], [ax, ay + 0.96, az]]
        molecules = [[0, 1, 2], [3, 4, 5]]
    return make_frame(symbols, positions, molecules)


class TestHbondsAt:

    def test_linear_bond_seen_from_both_sides(self):
        hb = hbonds_at([donor_water()], {"H2O": "O"})
        by_centre = dict(zip(hb.center, zip(hb.accepted, hb.donated)))
        assert by_centre == {0: (0, 1), 3: (1, 0)}
        assert hb.bonds[0] == frozenset({(0, 1, 3)})

    def test_bent_bond_rejected(self):
        # H1...O3 still 1.9 Å, but D-H...A = 120 deg < 150 deg.
        pos = (0.96 + 1.9 * np.cos(np.radians(60)), 1.9 * np.sin(np.radians(60)), 0.0)
        hb = hbonds_at([donor_water(acceptor_pos=pos)], {"H2O": "O"})
        assert hb.accepted.sum() == 0 and hb.donated.sum() == 0

    def test_distance_cutoff(self):
        hb = hbonds_at([donor_water()], {"H2O": "O"}, r_HA=1.8)
        assert hb.donated.sum() == 0

    def test_angle_cutoff_parameter(self):
        pos = (0.96 + 1.9 * np.cos(np.radians(60)), 1.9 * np.sin(np.radians(60)), 0.0)
        hb = hbonds_at([donor_water(acceptor_pos=pos)], {"H2O": "O"}, angle_min=110.0)
        assert hb.donated.sum() == 1

    def test_any_element_accepts(self):
        # A lone F atom as acceptor: counted, since partners can be any atom.
        hb = hbonds_at([donor_water("F", acceptor_h=False)], "H2O")
        assert list(hb.donated) == [1]

    def test_free_h_atom_never_donates(self):
        # Lone H 1.9 Å from the water O, pointing straight at it.
        f = make_frame(["O", "H", "H", "H"],
                       [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0], [-1.9, 0, 0]],
                       [[0, 1, 2], [3]])
        hb = hbonds_at([f], {"H2O": "O"})
        assert list(hb.accepted) == [0]

    def test_molecule_center_uses_all_its_atoms(self):
        hb = hbonds_at([donor_water()], "H2O")
        by_centre = dict(zip(hb.center, zip(hb.accepted, hb.donated)))
        assert by_centre == {0: (0, 1), 3: (1, 0)}

    def test_h_center_only_donates(self):
        hb = hbonds_at([donor_water()], "H")
        by_centre = dict(zip(hb.center, zip(hb.accepted, hb.donated)))
        assert by_centre[1] == (0, 1)
        assert sum(a for a, _ in by_centre.values()) == 0

    def test_frames_without_centre_give_no_samples(self):
        hb = hbonds_at([donor_water()], "HO")
        assert len(hb.accepted) == 0
        acc, don = hb.per_frame()
        assert np.isnan(acc).all() and np.isnan(don).all()

    def test_distribution_sums_to_one(self):
        frames = [donor_water(), zero_hbond_frame()]
        hb = hbonds_at(frames, {"H2O": "O"})
        P = hb.distribution()
        assert P.sum() == pytest.approx(1.0)
        assert P[1, 0] == pytest.approx(np.mean((hb.accepted == 1) & (hb.donated == 0)))

    def test_per_frame_averages_over_centres(self):
        acc, don = hbonds_at([donor_water()] * 3, {"H2O": "O"}).per_frame()
        np.testing.assert_allclose(acc, 0.5)
        np.testing.assert_allclose(don, 0.5)

    def test_acf_is_one_for_permanent_bond(self):
        lags, C = hbonds_at([donor_water()] * 20, {"H2O": "O"}).acf(max_lag=5)
        np.testing.assert_allclose(C, 1.0)

    def test_hbond_acf_unchanged_by_refactor(self):
        frames = [two_hbond_frame(), zero_hbond_frame()] * 10
        lags, C = hbond_acf(frames, max_lag=4)
        # bonds present every other frame: C = 1 at even lags, 0 at odd lags
        np.testing.assert_allclose(C, [0.0, 1.0, 0.0, 1.0])

    def test_trajectory_method_delegates(self):
        from chempiler import ChempilerTrajectory
        t = ChempilerTrajectory("dummy.traj")
        t.frames = [donor_water()]
        a = t.hbonds_at({"H2O": "O"})
        b = hbonds_at(t.frames, {"H2O": "O"})
        np.testing.assert_array_equal(a.accepted, b.accepted)


class TestHbondsAtCache:

    def _traj(self, n=3):
        from chempiler import ChempilerTrajectory
        t = ChempilerTrajectory("dummy.traj")
        t.frames = [donor_water(), zero_hbond_frame(), donor_water()][:n]
        return t

    def test_round_trip(self, tmp_path):
        t = self._traj()
        cache = tmp_path / "hb.json"
        computed = t.hbonds_at({"H2O": "O"}, cache_file=cache)
        assert cache.exists()
        loaded = t.hbonds_at({"H2O": "O"}, cache_file=cache)
        for k in ("frame", "center", "accepted", "donated"):
            np.testing.assert_array_equal(getattr(computed, k), getattr(loaded, k))
        assert computed.bonds == loaded.bonds
        np.testing.assert_allclose(computed.acf(max_lag=2)[1], loaded.acf(max_lag=2)[1])

    def test_different_parameters_raise(self, tmp_path):
        t = self._traj()
        cache = tmp_path / "hb.json"
        t.hbonds_at({"H2O": "O"}, cache_file=cache)
        with pytest.raises(ValueError, match="different"):
            t.hbonds_at({"H2O": "O"}, r_HA=2.0, cache_file=cache)
        with pytest.raises(ValueError, match="different"):
            self._traj(n=2).hbonds_at({"H2O": "O"}, cache_file=cache)
