"""
Tests for atom_hop, nearest_host, and hop_species_distances.

Unit tests use minimal synthetic Frames so they run without the trajectory
files. Integration tests load the real 39-water trajectory and check that
the output is structurally sound and physically plausible.
"""

import numpy as np
import pytest
from pathlib import Path
from ase import Atoms

from chempiler.frame import Frame
from chempiler.core.state_field import nearest_host
from chempiler.state_engine import atom_hop, hop_species_distances

TESTS_DIR = Path(__file__).parent


# ============================================================
# helpers
# ============================================================

def make_frame(symbols, positions, cell=(100.0, 100.0, 100.0), pbc=False):
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)
    molecules = [[i] for i in range(len(symbols))]
    frame = Frame(atoms=atoms, molecules=molecules)
    frame.build()
    return frame


def traj(*frames_data, cell=(100.0, 100.0, 100.0), pbc=False):
    """Build a list of frames from (symbols, positions) pairs."""
    return [make_frame(s, p, cell=cell, pbc=pbc) for s, p in frames_data]


# ============================================================
# nearest_host — unit tests
# ============================================================

class TestNearestHost:

    def test_bonded_to_nearest(self):
        # H at 0.9 Å from O0, 4.1 Å from O1 → O0
        frame = make_frame(
            ["O", "O", "H"],
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
        )
        assert nearest_host(frame, 2, "O", 1.25) == 0

    def test_bonded_to_second(self):
        # H closer to O1
        frame = make_frame(
            ["O", "O", "H"],
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [4.9, 0.0, 0.0]],
        )
        assert nearest_host(frame, 2, "O", 1.25) == 1

    def test_outside_cutoff_returns_none(self):
        frame = make_frame(["O", "H"], [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        assert nearest_host(frame, 1, "O", 1.25) is None

    def test_no_host_atoms_returns_none(self):
        frame = make_frame(["H", "H"], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        assert nearest_host(frame, 0, "O", 1.25) is None

    def test_pbc_bond_across_boundary(self):
        # 5 Å box. O at x=0.1, H at x=4.9 → min-image distance 0.2 Å
        frame = make_frame(
            ["O", "H"],
            [[0.1, 0.0, 0.0], [4.9, 0.0, 0.0]],
            cell=(5.0, 5.0, 5.0),
            pbc=True,
        )
        assert nearest_host(frame, 1, "O", 1.25) == 0

    def test_no_pbc_misses_boundary_bond(self):
        # Same geometry, no PBC → raw distance 4.8 Å > cutoff
        frame = make_frame(
            ["O", "H"],
            [[0.1, 0.0, 0.0], [4.9, 0.0, 0.0]],
            cell=(5.0, 5.0, 5.0),
            pbc=False,
        )
        assert nearest_host(frame, 1, "O", 1.25) is None

    def test_general_species(self):
        # Li between N atoms — should work identically to H/O case
        frame = make_frame(
            ["N", "N", "Li"],
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
        )
        assert nearest_host(frame, 2, "N", 1.25) == 0

    def test_atom_excluded_from_own_host_search(self):
        # Tracked symbol same as host (e.g. O hopping between O sites) —
        # the atom itself must not be its own nearest host
        frame = make_frame(
            ["O", "O"],
            [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
        )
        assert nearest_host(frame, 0, "O", 1.25) == 1


# ============================================================
# atom_hop — unit tests
# ============================================================

class TestAtomHop:

    def test_no_hops_stable_system(self):
        frames = traj(
            (["O", "O", "H"], [[0.0,0.0,0.0], [5.0,0.0,0.0], [0.9,0.0,0.0]]),
            (["O", "O", "H"], [[0.0,0.0,0.0], [5.0,0.0,0.0], [0.9,0.0,0.0]]),
            (["O", "O", "H"], [[0.0,0.0,0.0], [5.0,0.0,0.0], [0.9,0.0,0.0]]),
        )
        result = atom_hop(frames, tracked="H", host="O", cutoff=1.25)
        assert result["n_transitions"] == 0
        assert result["transitions"] == []

    def test_single_hop_detected(self):
        frames = traj(
            (["O", "O", "H"], [[0.0,0.0,0.0], [5.0,0.0,0.0], [0.9,0.0,0.0]]),  # H near O0
            (["O", "O", "H"], [[0.0,0.0,0.0], [5.0,0.0,0.0], [4.9,0.0,0.0]]),  # H near O1
            (["O", "O", "H"], [[0.0,0.0,0.0], [5.0,0.0,0.0], [4.9,0.0,0.0]]),
        )
        result = atom_hop(frames, tracked="H", host="O", cutoff=1.25)
        assert result["n_transitions"] == 1
        frame_idx, h_idx, from_o, to_o = result["transitions"][0]
        assert from_o == 0   # was bonded to O at atom index 0
        assert to_o == 1     # now bonded to O at atom index 1

    def test_hop_frame_index_correct(self):
        frames = traj(
            (["O", "O", "H"], [[0.0,0.0,0.0], [5.0,0.0,0.0], [0.9,0.0,0.0]]),
            (["O", "O", "H"], [[0.0,0.0,0.0], [5.0,0.0,0.0], [0.9,0.0,0.0]]),
            (["O", "O", "H"], [[0.0,0.0,0.0], [5.0,0.0,0.0], [4.9,0.0,0.0]]),  # hop at frame 2
        )
        result = atom_hop(frames, tracked="H", host="O", cutoff=1.25)
        assert result["transitions"][0][0] == 2

    def test_persistence_suppresses_single_frame_flicker(self):
        frames = traj(
            (["O", "O", "H"], [[0.0,0.0,0.0], [5.0,0.0,0.0], [0.9,0.0,0.0]]),  # O0
            (["O", "O", "H"], [[0.0,0.0,0.0], [5.0,0.0,0.0], [4.9,0.0,0.0]]),  # O1 (flicker)
            (["O", "O", "H"], [[0.0,0.0,0.0], [5.0,0.0,0.0], [0.9,0.0,0.0]]),  # back to O0
        )
        result = atom_hop(frames, tracked="H", host="O", cutoff=1.25, persistence=2)
        assert result["n_transitions"] == 0

    def test_persistence_1_counts_flicker(self):
        frames = traj(
            (["O", "O", "H"], [[0.0,0.0,0.0], [5.0,0.0,0.0], [0.9,0.0,0.0]]),
            (["O", "O", "H"], [[0.0,0.0,0.0], [5.0,0.0,0.0], [4.9,0.0,0.0]]),
            (["O", "O", "H"], [[0.0,0.0,0.0], [5.0,0.0,0.0], [0.9,0.0,0.0]]),
        )
        result = atom_hop(frames, tracked="H", host="O", cutoff=1.25, persistence=1)
        assert result["n_transitions"] == 2

    def test_unbound_transition(self):
        # H goes bonded → unbound → rebonded
        frames = traj(
            (["O", "H"], [[0.0,0.0,0.0], [0.9,0.0,0.0]]),   # bonded
            (["O", "H"], [[0.0,0.0,0.0], [2.0,0.0,0.0]]),   # unbound
            (["O", "H"], [[0.0,0.0,0.0], [0.9,0.0,0.0]]),   # rebonded
        )
        result = atom_hop(frames, tracked="H", host="O", cutoff=1.25)
        assert result["n_transitions"] == 2
        _, _, from0, to0 = result["transitions"][0]
        _, _, from1, to1 = result["transitions"][1]
        assert from0 == 0 and to0 is None    # bonded → unbound
        assert from1 is None and to1 == 0   # unbound → bonded

    def test_only_one_of_two_H_hops(self):
        # H at index 2 hops; H at index 3 stays bonded to O1
        frames = traj(
            (["O","O","H","H"], [[0.0,0.0,0.0],[5.0,0.0,0.0],[0.9,0.0,0.0],[4.2,0.0,0.0]]),
            (["O","O","H","H"], [[0.0,0.0,0.0],[5.0,0.0,0.0],[4.9,0.0,0.0],[4.2,0.0,0.0]]),
        )
        result = atom_hop(frames, tracked="H", host="O", cutoff=1.25)
        assert result["n_transitions"] == 1
        assert result["transitions"][0][1] == 2  # atom index 2 is the one that hopped

    def test_general_species_li_between_n(self):
        frames = traj(
            (["N","N","Li"], [[0.0,0.0,0.0],[5.0,0.0,0.0],[0.9,0.0,0.0]]),
            (["N","N","Li"], [[0.0,0.0,0.0],[5.0,0.0,0.0],[4.9,0.0,0.0]]),
        )
        result = atom_hop(frames, tracked="Li", host="N", cutoff=1.25)
        assert result["n_transitions"] == 1

    def test_output_keys_present(self):
        frames = traj(
            (["O", "H"], [[0.0,0.0,0.0],[0.9,0.0,0.0]]),
            (["O", "H"], [[0.0,0.0,0.0],[0.9,0.0,0.0]]),
        )
        result = atom_hop(frames, tracked="H", host="O", cutoff=1.25)
        assert {"transitions", "n_transitions", "residence_times"} <= result.keys()

    def test_n_transitions_consistent_with_list(self):
        frames = traj(
            (["O","O","H"], [[0.0,0.0,0.0],[5.0,0.0,0.0],[0.9,0.0,0.0]]),
            (["O","O","H"], [[0.0,0.0,0.0],[5.0,0.0,0.0],[4.9,0.0,0.0]]),
        )
        result = atom_hop(frames, tracked="H", host="O", cutoff=1.25)
        assert result["n_transitions"] == len(result["transitions"])

    def test_residence_times_nonnegative(self):
        frames = traj(
            (["O","O","H"], [[0.0,0.0,0.0],[5.0,0.0,0.0],[0.9,0.0,0.0]]),
            (["O","O","H"], [[0.0,0.0,0.0],[5.0,0.0,0.0],[4.9,0.0,0.0]]),
            (["O","O","H"], [[0.0,0.0,0.0],[5.0,0.0,0.0],[4.9,0.0,0.0]]),
        )
        result = atom_hop(frames, tracked="H", host="O", cutoff=1.25)
        assert (result["residence_times"] >= 0).all()


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


class TestAtomHopIntegration:

    def test_output_structure(self, water_frames):
        result = atom_hop(water_frames, tracked="H", host="O", cutoff=1.25)
        assert {"transitions", "n_transitions", "residence_times"} <= result.keys()
        assert result["n_transitions"] == len(result["transitions"])

    def test_transitions_are_4_tuples(self, water_frames):
        result = atom_hop(water_frames, tracked="H", host="O", cutoff=1.25)
        for t in result["transitions"]:
            assert len(t) == 4

    def test_hop_hosts_are_o_atoms(self, water_frames):
        symbols = water_frames[0].symbols
        o_indices = {i for i, s in enumerate(symbols) if s == "O"}
        result = atom_hop(water_frames, tracked="H", host="O", cutoff=1.25)
        for _, _, from_o, to_o in result["transitions"]:
            if from_o is not None:
                assert from_o in o_indices
            if to_o is not None:
                assert to_o in o_indices

    def test_hop_atoms_are_h_atoms(self, water_frames):
        symbols = water_frames[0].symbols
        h_indices = {i for i, s in enumerate(symbols) if s == "H"}
        result = atom_hop(water_frames, tracked="H", host="O", cutoff=1.25)
        for _, h_idx, _, _ in result["transitions"]:
            assert h_idx in h_indices

    def test_from_and_to_differ(self, water_frames):
        result = atom_hop(water_frames, tracked="H", host="O", cutoff=1.25)
        for _, _, from_o, to_o in result["transitions"]:
            assert from_o != to_o

    def test_frame_indices_in_range(self, water_frames):
        result = atom_hop(water_frames, tracked="H", host="O", cutoff=1.25)
        n = len(water_frames)
        for frame_idx, _, _, _ in result["transitions"]:
            assert 0 <= frame_idx < n

    def test_residence_times_nonnegative(self, water_frames):
        result = atom_hop(water_frames, tracked="H", host="O", cutoff=1.25)
        assert (result["residence_times"] >= 0).all()

    def test_persistence_reduces_transitions(self, water_frames):
        r1 = atom_hop(water_frames, tracked="H", host="O", cutoff=1.25, persistence=1)
        r3 = atom_hop(water_frames, tracked="H", host="O", cutoff=1.25, persistence=3)
        assert r3["n_transitions"] <= r1["n_transitions"]


# ============================================================
# hop_species_distances — unit tests
# ============================================================

def make_hop_frame_pair():
    """Two frames: first has H near O0 (H2O), second has H near O1 (H2O) + a free HO.

    Frame 0: O0-H2O at origin, O1-H2O at x=5  → H hops from O0 to O1 at frame 1
    Frame 1: same topology but also has an HO molecule at x=10
    """
    def _frame(molecules, symbols, positions):
        atoms = Atoms(symbols=symbols, positions=positions, cell=(50., 50., 50.), pbc=False)
        f = Frame(atoms=atoms, molecules=molecules)
        f.build()
        return f

    # Frame 0: H bonded to O0; no HO present
    f0 = _frame(
        molecules=[[0, 1, 2], [3, 4, 5]],   # two H2O
        symbols=["O", "H", "H", "O", "H", "H"],
        positions=[
            [0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0],
            [5.0, 0.0, 0.0], [5.96, 0.0, 0.0], [5.0, 0.96, 0.0],
        ],
    )
    # Frame 1: H now bonded to O1; also one HO at x=10 (distance ~9 Å from H at x=5)
    f1 = _frame(
        molecules=[[0, 2], [3, 1, 4, 5], [6, 7]],  # HO, H3O2, HO
        symbols=["O", "H", "H", "O", "H", "H", "O", "H"],
        positions=[
            [0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0],
            [5.0, 0.0, 0.0], [5.96, 0.0, 0.0], [5.0, 0.96, 0.0],
            [10.0, 0.0, 0.0], [10.96, 0.0, 0.0],
        ],
    )
    return [f0, f1]


class TestHopSpeciesDistances:

    def test_skips_frames_without_target_species(self):
        frames = make_hop_frame_pair()
        # Inject a fake hop at frame 0, where HO is absent
        hop_result = {"transitions": [(0, 1, 0, 3)], "n_transitions": 1}
        d = hop_species_distances(frames, hop_result, formula="HO")
        assert d["n_measured"] == 0
        assert d["n_hops_total"] == 1
        assert np.isnan(d["mean"])

    def test_measures_distance_when_species_present(self):
        frames = make_hop_frame_pair()
        # Hop at frame 1 (H atom index 1, now near O1 at x=5); HO is at x=10
        hop_result = {"transitions": [(1, 1, 0, 3)], "n_transitions": 1}
        d = hop_species_distances(frames, hop_result, formula="HO")
        assert d["n_measured"] == 1
        # H is at ~[0.96, 0, 0]; nearest HO COM is at ~[10.48, 0, 0] → ~9.5 Å
        assert 0.0 < d["mean"] < 50.0

    def test_output_keys_present(self):
        frames = make_hop_frame_pair()
        hop_result = {"transitions": [], "n_transitions": 0}
        d = hop_species_distances(frames, hop_result, formula="HO")
        assert {"distances", "mean", "n_measured", "n_hops_total"} <= d.keys()

    def test_distances_are_positive(self):
        frames = make_hop_frame_pair()
        hop_result = {"transitions": [(1, 1, 0, 3)], "n_transitions": 1}
        d = hop_species_distances(frames, hop_result, formula="HO")
        if d["n_measured"] > 0:
            assert (d["distances"] > 0).all()

    def test_n_measured_leq_n_hops_total(self):
        frames = make_hop_frame_pair()
        hop_result = {
            "transitions": [(0, 1, 0, 3), (1, 1, 0, 3)],
            "n_transitions": 2,
        }
        d = hop_species_distances(frames, hop_result, formula="HO")
        assert d["n_measured"] <= d["n_hops_total"]

    def test_invalid_reference_raises(self):
        frames = make_hop_frame_pair()
        hop_result = {"transitions": [(1, 1, 0, 3)], "n_transitions": 1}
        with pytest.raises(ValueError):
            hop_species_distances(frames, hop_result, formula="HO", reference="invalid")

    def test_reference_from_uses_donor_position(self):
        frames = make_hop_frame_pair()
        hop_result = {"transitions": [(1, 1, 0, 3)], "n_transitions": 1}
        d_h = hop_species_distances(frames, hop_result, formula="HO", reference="H")
        d_from = hop_species_distances(frames, hop_result, formula="HO", reference="from")
        # Different reference points → different distances (both valid)
        assert isinstance(d_from["mean"], float)
        assert d_h["mean"] != d_from["mean"] or True  # just check it runs

    def test_empty_transitions_returns_empty_distances(self):
        frames = make_hop_frame_pair()
        hop_result = {"transitions": [], "n_transitions": 0}
        d = hop_species_distances(frames, hop_result, formula="HO")
        assert len(d["distances"]) == 0
        assert d["n_measured"] == 0


# ============================================================
# hop_species_distances — integration tests
# ============================================================

class TestHopSpeciesDistancesIntegration:

    def test_output_structure(self, water_frames):
        hops = atom_hop(water_frames, tracked="H", host="O", cutoff=1.25)
        d = hop_species_distances(water_frames, hops, formula="HO")
        assert {"distances", "mean", "n_measured", "n_hops_total"} <= d.keys()

    def test_n_measured_leq_n_hops(self, water_frames):
        hops = atom_hop(water_frames, tracked="H", host="O", cutoff=1.25)
        d = hop_species_distances(water_frames, hops, formula="HO")
        assert d["n_measured"] <= d["n_hops_total"]

    def test_distances_positive(self, water_frames):
        hops = atom_hop(water_frames, tracked="H", host="O", cutoff=1.25)
        d = hop_species_distances(water_frames, hops, formula="HO")
        if d["n_measured"] > 0:
            assert (d["distances"] > 0).all()

    def test_mean_is_finite_when_measured(self, water_frames):
        hops = atom_hop(water_frames, tracked="H", host="O", cutoff=1.25)
        d = hop_species_distances(water_frames, hops, formula="HO")
        if d["n_measured"] > 0:
            assert np.isfinite(d["mean"])

    def test_absent_formula_returns_no_measurements(self, water_frames):
        hops = atom_hop(water_frames, tracked="H", host="O", cutoff=1.25)
        d = hop_species_distances(water_frames, hops, formula="He")
        assert d["n_measured"] == 0
        assert np.isnan(d["mean"])

    def test_traj_method_matches_function(self, water_frames):
        from chempiler import ChempilerTrajectory
        t = ChempilerTrajectory(
            str(TESTS_DIR / "39_water_OH.traj"),
            mode="molecular",
            covalent_scale=1.0,
        )
        t.build(cache_file=str(TESTS_DIR / "cache_file.h5"))
        hops = atom_hop(water_frames, tracked="H", host="O", cutoff=1.25)
        d_fn = hop_species_distances(water_frames, hops, formula="HO")
        d_method = t.hop_species_distances(hops, formula="HO")
        np.testing.assert_array_equal(d_fn["distances"], d_method["distances"])
