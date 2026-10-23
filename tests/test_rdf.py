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
from chempiler.rdf import rdf, rdf_shells

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
# error bars
# ============================================================

def random_pairs(n, seed=0):
    """n frames of two O atoms at independent random separations in [1, 3] Å."""
    rng = np.random.default_rng(seed)
    return [two_oxygens(d) for d in rng.uniform(1.0, 3.0, n)]


@pytest.mark.filterwarnings("ignore:RDF error bars may be too small")
class TestRDFErrors:

    def test_return_shapes(self):
        frames = random_pairs(50)
        out = rdf(frames, center="O", target="O", rmax=4.0, dr=0.1, errors=True)
        assert len(out) == 3
        out = rdf(frames, center="O", target="O", rmax=4.0, dr=0.1,
                  integrate=True, errors=True)
        assert len(out) == 5
        r, g, n, g_err, n_err = out
        assert g_err.shape == g.shape == n_err.shape == r.shape

    def test_g_unchanged_by_errors_flag(self):
        frames = random_pairs(50)
        r1, g1, n1 = rdf(frames, center="O", target="O", rmax=4.0, dr=0.1,
                         integrate=True)
        r2, g2, n2, _, _ = rdf(frames, center="O", target="O", rmax=4.0,
                               dr=0.1, integrate=True, errors=True)
        np.testing.assert_allclose(g1, g2)
        np.testing.assert_allclose(n1, n2)

    def test_identical_frames_give_zero_error(self):
        frames = [single_h2o()] * 20
        _, _, g_err = rdf(frames, center="O", target="H", rmax=3.0, dr=0.05,
                          errors=True, block_size=1)
        np.testing.assert_allclose(g_err, 0.0, atol=1e-12)

    def test_uncorrelated_matches_standard_error(self):
        # Constant normalisation: with blocks of one frame the error is the
        # plain standard error of the per-frame g(r).
        frames = random_pairs(200)
        r, g, g_err = rdf(frames, center="O", target="O", rmax=4.0, dr=0.1,
                          errors=True, block_size=1)
        per_frame = np.array([rdf([f], center="O", target="O", rmax=4.0,
                                  dr=0.1)[1] for f in frames])
        np.testing.assert_allclose(g_err, per_frame.std(axis=0, ddof=1) / np.sqrt(200))

    def test_block_means_give_error(self):
        # A fixed block size: the error is the spread of the block means of
        # the per-frame g(r) (Grossfield et al. Eq. 14).
        frames = random_pairs(200)
        r, g, g_err = rdf(frames, center="O", target="O", rmax=4.0, dr=0.1,
                          errors=True, block_size=10)
        per_frame = np.array([rdf([f], center="O", target="O", rmax=4.0,
                                  dr=0.1)[1] for f in frames])
        means = per_frame.reshape(20, 10, -1).mean(axis=1)
        np.testing.assert_allclose(g_err, means.std(axis=0, ddof=1) / np.sqrt(20))

    def test_repeated_frames_are_detected_as_correlated(self):
        # Each independent frame repeated k times: tau ~ k per bin, so the
        # estimated error must match that of the k-times shorter,
        # uncorrelated series -- not be sqrt(k) too small.
        k = 8
        unique = random_pairs(1000, seed=1)
        repeated = [f for f in unique for _ in range(k)]
        kw = dict(center="O", target="O", rmax=4.0, dr=0.1, errors=True)
        _, g, err_ref = rdf(unique, block_size=1, **kw)
        _, _, err_rep = rdf(repeated, **kw)
        populated = g > 0.5 * g.max()
        ratio = np.median(err_rep[populated] / err_ref[populated])
        assert 0.8 < ratio < 1.25, ratio

    def test_threads_match_serial_exactly(self):
        # Reproducibility: bitwise identical for any number of threads.
        frames = random_pairs(60)
        kw = dict(center="O", target="O", rmax=4.0, dr=0.1, integrate=True,
                  errors=True, return_neff=True)
        serial = rdf(frames, **kw)
        for workers in (2, 3, 7):
            for a, b in zip(serial, rdf(frames, n_workers=workers, **kw)):
                np.testing.assert_array_equal(a, b)

    def test_rerun_is_identical(self):
        frames = random_pairs(100, seed=3)
        kw = dict(center="O", target="O", rmax=4.0, dr=0.1, integrate=True,
                  errors=True, return_neff=True)
        for a, b in zip(rdf(frames, **kw), rdf(frames, **kw)):
            np.testing.assert_array_equal(a, b)

    def test_warns_when_no_plateau(self):
        # Runs of 10 identical frames, but with 200 frames and at least 20
        # blocks no block is longer than 8 frames: no plateau is reachable.
        frames = [f for f in random_pairs(20) for _ in range(10)]
        with pytest.warns(UserWarning, match="no plateau"):
            rdf(frames, center="O", target="O", rmax=4.0, dr=0.1, errors=True)

    def test_too_few_frames_gives_nan(self):
        with pytest.warns(UserWarning, match="NaN"):
            _, _, g_err = rdf(random_pairs(10), center="O", target="O",
                              rmax=4.0, dr=0.1, errors=True)
        assert np.isnan(g_err).all()

    def test_block_size_leaving_too_few_blocks_raises(self):
        with pytest.raises(ValueError, match="min_blocks"):
            rdf(random_pairs(100), center="O", target="O", rmax=4.0, dr=0.1,
                errors=True, block_size=10)

    def test_cache_round_trip(self, tmp_path):
        from chempiler import ChempilerTrajectory, load_rdf
        t = ChempilerTrajectory("dummy.traj")
        t.frames = random_pairs(50)
        kw = dict(center="O", target="O", rmax=4.0, dr=0.1, integrate=True,
                  errors=True, block_size=2)
        cache = tmp_path / "rdf.json"
        computed = t.rdf(cache_file=cache, **kw)
        loaded = t.rdf(cache_file=cache, **kw)
        assert len(loaded) == 5
        for a, b in zip(computed, loaded):
            np.testing.assert_allclose(a, b)
        for a, b in zip(computed, load_rdf(cache)):
            np.testing.assert_allclose(a, b)
        with pytest.raises(ValueError, match="different"):
            t.rdf(cache_file=cache, **{**kw, "errors": False})

    def test_cache_without_errors_keeps_old_key(self, tmp_path):
        # Caches written before error bars existed must still load.
        import json
        from chempiler import ChempilerTrajectory
        t = ChempilerTrajectory("dummy.traj")
        t.frames = random_pairs(20)
        cache = tmp_path / "rdf.json"
        t.rdf(cache_file=cache, center="O", target="O", rmax=4.0, dr=0.1)
        key = json.loads(cache.read_text())["param_key"]
        assert "errors" not in key and "block_size" not in key and "error_method" not in key

    def test_cache_from_other_error_estimator_raises(self, tmp_path):
        # Error bars from another estimator (e.g. the earlier
        # autocorrelation-time one) must never load as block-averaged ones.
        import json
        from chempiler import ChempilerTrajectory
        from chempiler.rdf import ERROR_METHOD
        t = ChempilerTrajectory("dummy.traj")
        t.frames = random_pairs(50)
        kw = dict(center="O", target="O", rmax=4.0, dr=0.1, errors=True)
        cache = tmp_path / "rdf.json"
        t.rdf(cache_file=cache, **kw)
        data = json.loads(cache.read_text())
        assert repr(ERROR_METHOD) in data["param_key"]
        data["param_key"] = data["param_key"].replace(repr(ERROR_METHOD), repr("sokal"))
        cache.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="error estimator"):
            t.rdf(cache_file=cache, **kw)

    def test_neff_return_shapes(self):
        frames = random_pairs(50)
        out = rdf(frames, center="O", target="O", rmax=4.0, dr=0.1,
                  errors=True, return_neff=True)
        assert len(out) == 4
        out = rdf(frames, center="O", target="O", rmax=4.0, dr=0.1,
                  integrate=True, errors=True, return_neff=True)
        assert len(out) == 7
        r, g, n, g_err, n_err, g_neff, n_neff = out
        assert g_neff.shape == n_neff.shape == r.shape

    def test_neff_requires_errors(self):
        with pytest.raises(ValueError, match="requires errors"):
            rdf(random_pairs(5), center="O", target="O", rmax=4.0, dr=0.1,
                return_neff=True)

    def test_neff_is_number_of_blocks(self):
        frames = random_pairs(100)
        _, _, _, _, _, g_neff, n_neff = rdf(
            frames, center="O", target="O", rmax=4.0, dr=0.1, integrate=True,
            errors=True, block_size=4, return_neff=True)
        np.testing.assert_allclose(g_neff, 25.0)
        np.testing.assert_allclose(n_neff, 25.0)

    def test_repeated_frames_get_blocks_at_least_as_long_as_runs(self):
        # k copies of each independent frame: blocking must pick blocks of
        # at least k frames, so no two blocks share a run.
        k, n_unique = 8, 1000
        unique = random_pairs(n_unique, seed=1)
        repeated = [f for f in unique for _ in range(k)]
        _, g, _, g_neff = rdf(repeated, center="O", target="O", rmax=4.0,
                              dr=0.1, errors=True, return_neff=True)
        assert len(repeated) // g_neff[0] >= k

    def test_cache_round_trip_with_neff(self, tmp_path):
        from chempiler import ChempilerTrajectory, load_rdf
        t = ChempilerTrajectory("dummy.traj")
        t.frames = random_pairs(50)
        kw = dict(center="O", target="O", rmax=4.0, dr=0.1, integrate=True,
                  errors=True, block_size=2, return_neff=True)
        cache = tmp_path / "rdf.json"
        computed = t.rdf(cache_file=cache, **kw)
        loaded = t.rdf(cache_file=cache, **kw)
        assert len(computed) == len(loaded) == len(load_rdf(cache)) == 7
        for a, b in zip(computed, loaded):
            np.testing.assert_allclose(a, b)

    def test_tau_return_order(self):
        out = rdf(random_pairs(100), center="O", target="O", rmax=4.0, dr=0.1,
                  integrate=True, errors=True, return_neff=True, return_tau=True)
        assert len(out) == 9
        r, g_tau, n_tau = out[0], out[7], out[8]
        assert g_tau.shape == n_tau.shape == r.shape
        out = rdf(random_pairs(100), center="O", target="O", rmax=4.0, dr=0.1,
                  errors=True, return_tau=True)
        assert len(out) == 4

    def test_tau_requires_errors(self):
        with pytest.raises(ValueError, match="requires errors"):
            rdf(random_pairs(30), center="O", target="O", rmax=4.0, dr=0.1,
                return_tau=True)

    def test_tau_is_one_for_single_frame_blocks(self):
        # Blocks of one frame: the error is the plain standard error, so the
        # implied correlation time is exactly 1 wherever the bin fluctuates.
        r, g, g_err, g_tau = rdf(random_pairs(200), center="O", target="O",
                                 rmax=4.0, dr=0.1, errors=True, block_size=1,
                                 return_tau=True)
        np.testing.assert_allclose(g_tau[g_err > 0], 1.0)

    def test_tau_counts_repeated_frames(self):
        # k copies of each independent frame: tau ~ k in populated bins.
        k = 8
        repeated = [f for f in random_pairs(1000, seed=1) for _ in range(k)]
        _, g, _, g_tau = rdf(repeated, center="O", target="O", rmax=4.0, dr=0.1,
                             errors=True, return_tau=True)
        populated = g > 0.5 * g.max()
        assert 0.75 * k < np.median(g_tau[populated]) < 1.25 * k

    def test_cache_round_trip_with_tau(self, tmp_path):
        import json
        from chempiler import ChempilerTrajectory, load_rdf
        t = ChempilerTrajectory("dummy.traj")
        t.frames = random_pairs(60)
        kw = dict(center="O", target="O", rmax=4.0, dr=0.1, integrate=True,
                  errors=True, block_size=2, return_tau=True)
        cache = tmp_path / "rdf.json"
        computed = t.rdf(cache_file=cache, **kw)
        loaded = t.rdf(cache_file=cache, **kw)
        assert len(computed) == len(loaded) == len(load_rdf(cache)) == 7
        for a, b in zip(computed, loaded):
            np.testing.assert_allclose(a, b)
        # a cache without tau keeps its old key
        t.rdf(cache_file=tmp_path / "plain.json", **{**kw, "return_tau": False})
        assert "return_tau" not in json.loads((tmp_path / "plain.json").read_text())["param_key"]

    def test_cache_without_neff_keeps_old_key(self, tmp_path):
        # Error-bar caches written before return_neff existed must still load.
        import json
        from chempiler import ChempilerTrajectory
        t = ChempilerTrajectory("dummy.traj")
        t.frames = random_pairs(40)
        cache = tmp_path / "rdf.json"
        t.rdf(cache_file=cache, center="O", target="O", rmax=4.0, dr=0.1,
              errors=True, block_size=2)
        key = json.loads(cache.read_text())["param_key"]
        assert "errors" in key and "return_neff" not in key


# ============================================================
# target=None: every atom
# ============================================================

class TestRDFTargetNone:

    def test_matches_element_target_when_one_element(self):
        frames = [two_oxygens(1.5), two_oxygens(2.5)]
        r1, g1, n1 = rdf(frames, center="O", target="O", rmax=4.0, dr=0.1, integrate=True)
        r2, g2, n2 = rdf(frames, center="O", target=None, rmax=4.0, dr=0.1, integrate=True)
        np.testing.assert_allclose(g1, g2)
        np.testing.assert_allclose(n1, n2)

    def test_atom_center_counts_own_molecule_but_not_itself(self):
        # O of one H2O: the two H at 0.96 Å count, the O itself does not.
        r, g, n = rdf([single_h2o()], center="O", target=None, rmax=3.0, dr=0.05,
                      integrate=True)
        assert n[-1] == pytest.approx(2.0)
        assert n[r < 0.9].max() == pytest.approx(0.0)

    def test_molecule_center_excludes_own_atoms(self):
        # COM of each of two waters 3 Å apart: only the other water's 3 atoms count.
        w = make_frame(
            symbols=["O", "H", "H", "O", "H", "H"],
            positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0],
                       [3, 0, 0], [3.96, 0, 0], [3, 0.96, 0]],
            molecules=[[0, 1, 2], [3, 4, 5]],
        )
        r, g, n = rdf([w], center="H2O", target=None, rmax=6.0, dr=0.05, integrate=True)
        assert n[-1] == pytest.approx(3.0)
        assert n[r < 1.5].max() == pytest.approx(0.0)


# ============================================================
# rmax vs minimum-image limit
# ============================================================

class TestRmaxLimit:

    def test_warns_beyond_half_box(self):
        frames = [two_oxygens(1.5, cell=(10., 10., 10.))]
        with pytest.warns(UserWarning, match="half the smallest cell width"):
            rdf(frames, center="O", target="O", rmax=6.0, dr=0.1)

    def test_no_warning_within_half_box(self, recwarn):
        frames = [two_oxygens(1.5, cell=(10., 10., 10.))]
        rdf(frames, center="O", target="O", rmax=5.0, dr=0.1)
        assert not [w for w in recwarn if "cell width" in str(w.message)]

    def test_uses_smallest_width_over_frames(self):
        frames = [two_oxygens(1.5, cell=(12., 12., 12.)),
                  two_oxygens(1.5, cell=(10., 12., 12.))]
        with pytest.warns(UserWarning, match="5.000 Å"):
            rdf(frames, center="O", target="O", rmax=5.5, dr=0.1)

    def test_default_rmax_is_half_smallest_width(self):
        frames = [two_oxygens(1.5, cell=(10., 12., 14.))]
        r, _ = rdf(frames, center="O", target="O", dr=0.1)
        assert r[-1] < 5.0 + 0.1

    def test_skewed_cell_uses_perpendicular_width(self):
        # Face separation along a is 10*sin(60°) = 8.66, not 10.
        cell = [[10., 0., 0.], [5., 8.660254, 0.], [0., 0., 10.]]
        frames = [two_oxygens(1.5, cell=cell)]
        with pytest.warns(UserWarning, match="4.330 Å"):
            rdf(frames, center="O", target="O", rmax=4.8, dr=0.1)


# ============================================================
# shell detection
# ============================================================

def model_gr(r, peaks, width=0.12, tail=None):
    """Sum of Gaussian peaks on a baseline that is 0 at small r, 1 beyond."""
    g = sum(h * np.exp(-0.5 * ((r - p) / width) ** 2) for p, h in peaks)
    base = 1.0 / (1.0 + np.exp(-(r - (tail or peaks[-1][0] + 0.8)) / 0.1))
    return g + base


class TestRDFShells:

    r = np.arange(0.01, 6.0, 0.02)

    def test_minimum_between_peaks_is_refined(self):
        g = model_gr(self.r, [(1.0, 3.0), (2.8, 1.5)])
        s = rdf_shells(self.r, g, smooth_sigma=0.0, refine_width=0.1)
        assert len(s.peaks) == 2
        # the minimum between the two peaks is found and lies between them
        assert 1.2 < s.troughs[1] < 2.6
        assert s.g_min[0] == pytest.approx(np.interp(s.troughs[1], self.r, g))

    def test_open_last_shell_gives_nan_cn(self):
        # g(r) falls after the last peak and never rises again before rmax.
        r = self.r
        g = 2.0 * np.exp(-0.5 * ((r - 2.8) / 0.2) ** 2)
        n = np.cumsum(g) * 0.02
        s = rdf_shells(r, g, n, smooth_sigma=0.0)
        assert len(s.peaks) == 1
        assert s.troughs[-1] == pytest.approx(r[-1])
        assert np.isnan(s.cns[-1]) and np.isnan(s.g_min[-1]) and np.isnan(s.cn_spread[-1])

    def test_closed_last_shell_gets_cn(self):
        g = model_gr(self.r, [(2.8, 2.5)], tail=3.6)
        n = np.cumsum(g) * 0.02
        s = rdf_shells(self.r, g, n, smooth_sigma=0.1)
        assert len(s.peaks) == 1
        assert s.troughs[-1] < self.r[-1] - 0.5
        assert s.cns[0] == pytest.approx(np.interp(s.troughs[1], self.r, n))
        assert s.cn_spread[0] > 0

    def test_smoothing_does_not_create_peak_at_rmax(self):
        # Rising tail up to rmax: zero-padding used to pull it down into a peak.
        r = self.r
        g = model_gr(r, [(2.8, 2.5)], tail=3.6) + 0.05 * (r - 3.6).clip(0)
        s = rdf_shells(r, g, smooth_sigma=0.3)
        assert (s.peaks < r[-1] - 1.0).all(), s.peaks

    def test_without_n_cns_are_none(self):
        g = model_gr(self.r, [(1.0, 3.0), (2.8, 1.5)])
        s = rdf_shells(self.r, g)
        assert s.cns is None and s.cn_spread is None
        assert len(s.g_min) == len(s.peaks)


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


# ============================================================
# window_counts
# ============================================================

@pytest.mark.filterwarnings("ignore:.*error bars may be too small")
@pytest.mark.filterwarnings("ignore:.*error bars are NaN")
class TestWindowCounts:

    def test_window_from_zero_matches_rdf(self):
        from chempiler.rdf import window_counts
        frames = random_pairs(200)
        r, g, n, g_err, n_err = rdf(frames, center="O", target="O", rmax=4.0,
                                    dr=0.1, integrate=True, errors=True, block_size=1)
        r_hi = 2.0                      # a bin edge: n at bin centre 1.95 covers [0, 2.0)
        i = int(np.argmin(abs(r - 1.95)))
        cn, cn_err, neff = window_counts(frames, "O", "O", [(0.0, r_hi)], block_size=1)
        np.testing.assert_allclose(cn[0], n[i])
        np.testing.assert_allclose(cn_err[0], n_err[i])
        np.testing.assert_allclose(neff, 200.0)

    def test_adjacent_windows_add_up(self):
        from chempiler.rdf import window_counts
        frames = random_pairs(100)
        cn, _, _ = window_counts(frames, "O", "O", [(1.0, 2.0), (2.0, 2.5), (1.0, 2.5)])
        np.testing.assert_allclose(cn[0] + cn[1], cn[2])

    def test_constant_count_has_zero_error(self):
        from chempiler.rdf import window_counts
        frames = random_pairs(50)          # every separation lies in [1, 3) Å
        cn, cn_err, _ = window_counts(frames, "O", "O", [(0.5, 3.5)], block_size=1)
        np.testing.assert_allclose(cn, 1.0)   # each O centre has exactly one partner
        np.testing.assert_allclose(cn_err, 0.0, atol=1e-12)

    def test_series_is_instantaneous_count(self):
        from chempiler.rdf import window_counts
        frames = [two_oxygens(d) for d in (1.5, 2.5, 1.5, 3.5)]
        cn, _, _, series = window_counts(frames, "O", "O", [(1.0, 2.0), (1.0, 3.0)],
                                         return_series=True)
        np.testing.assert_allclose(series, [[1, 1], [0, 1], [1, 1], [0, 0]])
        np.testing.assert_allclose(series.mean(axis=0), cn)   # constant volume

    def test_series_cache_round_trip(self, tmp_path):
        from chempiler import ChempilerTrajectory
        t = ChempilerTrajectory("dummy.traj")
        t.frames = random_pairs(60)
        cache = tmp_path / "cn.json"
        kw = dict(center="O", target="O", windows=[(1.0, 2.0)], block_size=2,
                  return_series=True)
        computed = t.window_counts(cache_file=cache, **kw)
        loaded = t.window_counts(cache_file=cache, **kw)
        assert len(computed) == len(loaded) == 4
        for a, b in zip(computed, loaded):
            np.testing.assert_allclose(a, b)

    def test_bad_window_raises(self):
        from chempiler.rdf import window_counts
        with pytest.raises(ValueError, match="r_lo < r_hi"):
            window_counts(random_pairs(5), "O", "O", [(2.0, 1.0)])

    def test_cache_round_trip(self, tmp_path):
        from chempiler import ChempilerTrajectory
        t = ChempilerTrajectory("dummy.traj")
        t.frames = random_pairs(50)
        cache = tmp_path / "cn.json"
        kw = dict(center="O", target="O", windows=[(1.0, 2.0), (2.0, 3.0)], block_size=2)
        computed = t.window_counts(cache_file=cache, **kw)
        loaded = t.window_counts(cache_file=cache, **kw)
        for a, b in zip(computed, loaded):
            np.testing.assert_allclose(a, b)
        with pytest.raises(ValueError, match="different"):
            t.window_counts(cache_file=cache, **{**kw, "windows": [(1.0, 2.5)]})
