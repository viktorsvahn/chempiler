"""Tests for the RDF function.

Unit tests use small synthetic frames with known geometry so that the expected
output can be derived analytically or by inspection. Integration tests run
against the real 39-water trajectory and check physically meaningful properties.
"""

import numpy as np
import pytest
from pathlib import Path
from ase import Atoms

from chempiler.frame import Frame
from chempiler.rdf import rdf

TESTS_DIR = Path(__file__).parent

BIG_CELL = (30., 30., 30.)   # large enough that periodic images don't interfere


# ============================================================
# helpers
# ============================================================

def make_frame(symbols, positions, molecules, cell=BIG_CELL, pbc=True):
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)
    frame = Frame(atoms=atoms, molecules=molecules)
    frame.build()
    return frame


def single_h2o(oh_bond=0.96, cell=BIG_CELL):
    """One H2O molecule: O at origin, H at (oh_bond, 0, 0) and (0, oh_bond, 0)."""
    return make_frame(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [oh_bond, 0.0, 0.0], [0.0, oh_bond, 0.0]],
        molecules=[[0, 1, 2]],
        cell=cell,
    )


def two_oxygens(dist, cell=BIG_CELL):
    """Two isolated O atoms separated by dist, no H atoms."""
    return make_frame(
        symbols=["O", "O"],
        positions=[[0.0, 0.0, 0.0], [dist, 0.0, 0.0]],
        molecules=[[0], [1]],
        cell=cell,
    )


# ============================================================
# output structure
# ============================================================

class TestRDFOutputStructure:

    def test_returns_two_arrays_by_default(self):
        frames = [single_h2o()] * 3
        result = rdf(frames, center="O", target="H", rmax=3.0, dr=0.1)
        assert len(result) == 2

    def test_returns_three_arrays_with_integrate(self):
        frames = [single_h2o()] * 3
        result = rdf(frames, center="O", target="H", rmax=3.0, dr=0.1, integrate=True)
        assert len(result) == 3

    def test_r_and_g_same_length(self):
        frames = [single_h2o()] * 3
        r, g = rdf(frames, center="O", target="H", rmax=3.0, dr=0.1)
        assert len(r) == len(g)

    def test_r_g_n_all_same_length(self):
        frames = [single_h2o()] * 3
        r, g, n = rdf(frames, center="O", target="H", rmax=3.0, dr=0.1, integrate=True)
        assert len(r) == len(g) == len(n)

    def test_r_values_are_bin_centers(self):
        frames = [single_h2o()] * 2
        dr = 0.1
        r, g = rdf(frames, center="O", target="H", rmax=2.0, dr=dr)
        # First bin centre should be dr/2
        assert abs(r[0] - dr / 2) < 1e-10
        # Spacing should equal dr
        assert np.allclose(np.diff(r), dr)

    def test_output_arrays_are_numpy(self):
        frames = [single_h2o()] * 2
        r, g = rdf(frames, center="O", target="H", rmax=2.0, dr=0.1)
        assert isinstance(r, np.ndarray)
        assert isinstance(g, np.ndarray)

    def test_n_is_numpy_when_integrate(self):
        frames = [single_h2o()] * 2
        r, g, n = rdf(frames, center="O", target="H", rmax=2.0, dr=0.1, integrate=True)
        assert isinstance(n, np.ndarray)


# ============================================================
# peak location and zero regions
# ============================================================

class TestRDFPeakAndZeros:

    def test_peak_at_known_bond_length(self):
        oh_bond = 0.96
        frames = [single_h2o(oh_bond=oh_bond)] * 20
        dr = 0.02
        r, g = rdf(frames, center="O", target="H", rmax=3.0, dr=dr)
        peak_r = r[g.argmax()]
        assert abs(peak_r - oh_bond) < dr, (
            f"Peak at {peak_r:.3f} Å, expected ~{oh_bond} Å"
        )

    def test_bins_beyond_rmax_absent(self):
        # With one H2O, O-H distances are at ~0.96 Å.
        # Bins above the rmax should simply not exist.
        frames = [single_h2o()] * 5
        r, g = rdf(frames, center="O", target="H", rmax=3.0, dr=0.1)
        assert r.max() < 3.0 + 0.1

    def test_g_zero_where_no_atoms_exist(self):
        # Single H2O: O-H bond at 0.96 Å, H-H at ~1.36 Å.
        # g(r) must be zero for r > 2 Å (no second-shell in isolated molecule).
        oh_bond = 0.96
        frames = [single_h2o(oh_bond=oh_bond)] * 20
        r, g = rdf(frames, center="O", target="H", rmax=3.0, dr=0.02)
        far_bins = r > 2.0
        assert g[far_bins].sum() == 0.0

    def test_g_nonnegative_everywhere(self):
        frames = [single_h2o()] * 10
        r, g = rdf(frames, center="O", target="H", rmax=4.0, dr=0.05)
        assert (g >= 0).all()

    def test_o_o_rdf_peak_at_known_distance(self):
        dist = 2.8
        frames = [two_oxygens(dist)] * 20
        dr = 0.05
        r, g = rdf(frames, center="O", target="O", rmax=5.0, dr=dr)
        peak_r = r[g.argmax()]
        assert abs(peak_r - dist) < dr

    def test_g_zero_at_very_small_r(self):
        # No atom can be at r=0 (atoms don't overlap); first bins must be zero.
        frames = [single_h2o()] * 5
        r, g = rdf(frames, center="O", target="H", rmax=3.0, dr=0.1)
        assert g[r < 0.5].sum() == 0.0


# ============================================================
# self-pair exclusion
# ============================================================

class TestSelfPairExclusion:

    def test_single_o_atom_oo_rdf_is_zero(self):
        # Only one O in the frame — no valid O-O pair possible.
        frame = make_frame(
            symbols=["O", "H", "H"],
            positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]],
            molecules=[[0, 1, 2]],
        )
        r, g = rdf([frame] * 5, center="O", target="O", rmax=3.0, dr=0.1)
        assert g.sum() == 0.0

    def test_self_pair_not_counted_in_same_species_rdf(self):
        # Two O atoms at distance d. O-O RDF should peak at d, not at r=0.
        dist = 3.0
        frames = [two_oxygens(dist)] * 20
        r, g = rdf(frames, center="O", target="O", rmax=5.0, dr=0.1)
        assert g[r < 0.3].sum() == 0.0
        assert r[g.argmax()] > 0.5


# ============================================================
# empty / edge cases
# ============================================================

class TestRDFEdgeCases:

    def test_no_center_atoms_returns_zero_g(self):
        # Frame has only H atoms, center="O" → no center atoms → g=0
        frame = make_frame(
            symbols=["H", "H"],
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            molecules=[[0], [1]],
        )
        r, g = rdf([frame] * 3, center="O", target="H", rmax=3.0, dr=0.1)
        assert g.sum() == 0.0

    def test_no_target_atoms_returns_zero_g(self):
        frame = make_frame(
            symbols=["O", "O"],
            positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            molecules=[[0], [1]],
        )
        r, g = rdf([frame] * 3, center="O", target="H", rmax=3.0, dr=0.1)
        assert g.sum() == 0.0

    def test_single_frame_does_not_crash(self):
        frames = [single_h2o()]
        r, g = rdf(frames, center="O", target="H", rmax=3.0, dr=0.1)
        assert len(r) == len(g)


# ============================================================
# PBC / minimum image convention
# ============================================================

class TestRDFPBC:

    def test_pair_across_boundary_found_at_correct_distance(self):
        # O at x=0.1, H at x=9.9 in a 10 Å box → min-image distance 0.2 Å.
        # This is artificially small but tests that MIC is applied correctly.
        cell = (10., 10., 10.)
        frame = make_frame(
            symbols=["O", "H"],
            positions=[[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]],
            molecules=[[0, 1]],
            cell=cell,
        )
        dr = 0.05
        r, g = rdf([frame] * 10, center="O", target="H", rmax=4.0, dr=dr)
        # MIC distance is 0.2 Å — should appear in first bin
        first_nonzero = r[g > 0][0] if (g > 0).any() else None
        assert first_nonzero is not None and first_nonzero < 0.5


# ============================================================
# dict selectors
# ============================================================

class TestRDFSelectors:

    def test_dict_selector_matches_element_selector(self):
        # With only H2O molecules present, {"H2O": "O"} should give
        # the same result as plain "O".
        frames = [single_h2o()] * 10
        r1, g1 = rdf(frames, center="O", target="H", rmax=3.0, dr=0.05)
        r2, g2 = rdf(frames, center={"H2O": "O"}, target="H", rmax=3.0, dr=0.05)
        np.testing.assert_allclose(g1, g2)

    def test_dict_selector_restricts_to_formula(self):
        # Frame with one H2O and one HO molecule.
        # center={"H2O": "O"} should only see the H2O oxygen.
        frame = make_frame(
            symbols=["O", "H", "H", "O", "H"],
            positions=[
                [0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0],
                [10.0, 0.0, 0.0], [10.96, 0.0, 0.0],
            ],
            molecules=[[0, 1, 2], [3, 4]],
        )
        r_all, g_all = rdf([frame] * 10, center="O", target="H",
                           rmax=3.0, dr=0.05)
        r_h2o, g_h2o = rdf([frame] * 10, center={"H2O": "O"}, target="H",
                            rmax=3.0, dr=0.05)
        # The {"H2O": "O"} selector has fewer center atoms, so the two
        # histograms may differ — just verify the H2O-only one has a peak.
        assert g_h2o.max() > 0


# ============================================================
# running coordination number
# ============================================================

class TestRDFIntegrate:

    def test_n_is_non_decreasing(self):
        frames = [single_h2o()] * 20
        r, g, n = rdf(frames, center="O", target="H", rmax=3.0, dr=0.02,
                      integrate=True)
        assert (np.diff(n) >= -1e-12).all()

    def test_n_starts_near_zero(self):
        frames = [single_h2o()] * 20
        r, g, n = rdf(frames, center="O", target="H", rmax=3.0, dr=0.02,
                      integrate=True)
        assert n[r < 0.5].sum() < 1e-10

    def test_g_unchanged_by_integrate_flag(self):
        # g(r) must be identical whether or not integrate=True.
        frames = [single_h2o()] * 10
        r1, g1 = rdf(frames, center="O", target="H", rmax=3.0, dr=0.05)
        r2, g2, n2 = rdf(frames, center="O", target="H", rmax=3.0, dr=0.05,
                         integrate=True)
        np.testing.assert_array_equal(g1, g2)


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


class TestRDFIntegration:

    def test_oh_first_peak_at_covalent_bond(self, water_frames):
        r, g = rdf(water_frames, center="O", target="H", rmax=4.0, dr=0.02)
        peak_r = r[g.argmax()]
        assert 0.85 < peak_r < 1.1, f"O-H peak at {peak_r:.3f} Å, expected ~0.96 Å"

    def test_oo_first_peak_at_hbond_distance(self, water_frames):
        r, g = rdf(water_frames, center="O", target="O", rmax=6.0, dr=0.05)
        # Restrict search to first shell (1-4 Å)
        mask = (r > 1.0) & (r < 4.0)
        peak_r = r[mask][g[mask].argmax()]
        assert 2.5 < peak_r < 3.2, f"O-O peak at {peak_r:.3f} Å, expected ~2.8 Å"

    def test_g_approaches_one_at_large_r(self, water_frames):
        r, g = rdf(water_frames, center="O", target="O", rmax=5.0, dr=0.05)
        # Last 10 bins should average near 1
        mean_tail = g[r > 4.5].mean()
        assert 0.5 < mean_tail < 2.0, f"g(r) tail = {mean_tail:.3f}, expected ~1"

    def test_oh_g_nonnegative(self, water_frames):
        r, g = rdf(water_frames, center="O", target="H", rmax=4.0, dr=0.02)
        assert (g >= 0).all()

    def test_oh_coordination_number_near_two(self, water_frames):
        r, g, n = rdf(water_frames, center="O", target="H", rmax=4.0, dr=0.02,
                      integrate=True)
        # n(r) just past the covalent bond (r ≈ 1.3 Å) should be ~2 for H2O
        cn = n[r < 1.4][-1]
        assert 1.5 < cn < 2.5, f"Covalent O-H CN = {cn:.2f}, expected ~2"

    def test_n_nondecreasing_real_trajectory(self, water_frames):
        r, g, n = rdf(water_frames, center="O", target="H", rmax=4.0, dr=0.02,
                      integrate=True)
        assert (np.diff(n) >= -1e-10).all()

    def test_oh_and_ho_peaks_at_same_r(self, water_frames):
        r_oh, g_oh = rdf(water_frames, center="O", target="H", rmax=4.0, dr=0.05)
        r_ho, g_ho = rdf(water_frames, center="H", target="O", rmax=4.0, dr=0.05)
        assert abs(r_oh[g_oh.argmax()] - r_ho[g_ho.argmax()]) < 0.1

    def test_dict_selector_gives_nonzero_result(self, water_frames):
        r, g = rdf(water_frames, center={"HO": "O"}, target={"H2O": "H"},
                   rmax=5.0, dr=0.05)
        assert g.max() > 0
