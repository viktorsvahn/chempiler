"""Tests for block-averaging statistics (chempiler.core.statistics)."""

import numpy as np
import pytest

from chempiler.core.statistics import (split_blocks, block_average,
                                       estimate_autocorrelation_time,
                                       integrated_autocorrelation_times,
                                       confidence_halfwidth, blocking_levels,
                                       plateau_level, blocked_stderr)


# ============================================================
# split_blocks
# ============================================================

class TestSplitBlocks:

    def test_output_shape(self):
        blocks = split_blocks(100, 10)
        assert blocks.shape == (10, 10)

    def test_block_size_respected(self):
        blocks = split_blocks(60, 20)
        assert blocks.shape[1] == 20

    def test_indices_are_contiguous(self):
        blocks = split_blocks(30, 5)
        flat = blocks.flatten()
        # Must be 0, 1, 2, ..., 29 (first n_blocks*block_size indices)
        np.testing.assert_array_equal(flat, np.arange(30))

    def test_non_divisible_input_raises(self):
        # split_blocks uses reshape; 25 cannot reshape into (2, 10)
        with pytest.raises((ValueError, Exception)):
            split_blocks(25, 10)

    def test_exactly_two_blocks(self):
        blocks = split_blocks(20, 10)
        assert blocks.shape[0] == 2

    def test_fewer_than_two_blocks_raises(self):
        with pytest.raises(ValueError):
            split_blocks(5, 10)   # only 0 complete blocks

    def test_one_block_raises(self):
        with pytest.raises(ValueError):
            split_blocks(10, 10)  # exactly 1 block < 2

    def test_n_blocks_large_tau_corr_raises(self):
        with pytest.raises(ValueError):
            split_blocks(100, 60)  # 100//60 = 1 block


# ============================================================
# block_average
# ============================================================

class TestBlockAverage:

    def test_constant_series_stderr_zero(self):
        values = np.ones(100)
        result = block_average(values, 10)
        assert abs(result["stderr"]) < 1e-12

    def test_constant_series_mean_correct(self):
        values = np.full(100, 5.0)
        result = block_average(values, 10)
        assert abs(result["mean"] - 5.0) < 1e-12

    def test_mean_of_linear_series(self):
        # Values 0..99 → mean = 49.5
        values = np.arange(100, dtype=float)
        result = block_average(values, 10)
        assert abs(result["mean"] - 49.5) < 1e-10

    def test_n_blocks_in_result(self):
        values = np.ones(60)
        result = block_average(values, 10)
        assert result["n_blocks"] == 6

    def test_result_has_required_keys(self):
        result = block_average(np.ones(20), 10)
        assert {"mean", "stderr", "n_blocks"} <= result.keys()

    def test_stderr_non_negative(self):
        rng = np.random.default_rng(42)
        values = rng.standard_normal(100)
        result = block_average(values, 10)
        assert result["stderr"] >= 0.0

    def test_larger_blocks_same_mean(self):
        # Block size does not affect the mean of a constant series
        values = np.full(100, 3.0)
        r1 = block_average(values, 5)
        r2 = block_average(values, 10)
        assert abs(r1["mean"] - r2["mean"]) < 1e-12

    def test_too_few_blocks_raises(self):
        with pytest.raises(ValueError):
            block_average(np.ones(5), 10)

    def test_known_two_block_stderr(self):
        # Two blocks with means 0 and 2 → overall mean 1, stderr = std([0,2])/sqrt(2)
        # std([0,2], ddof=1) = sqrt(2), stderr = sqrt(2)/sqrt(2) = 1.0
        values = np.array([0.0, 0.0, 2.0, 2.0])  # block_size=2, 2 blocks
        result = block_average(values, 2)
        assert abs(result["mean"] - 1.0) < 1e-10
        assert abs(result["stderr"] - 1.0) < 1e-10


# ============================================================
# integrated_autocorrelation_times
# ============================================================

class TestIntegratedAutocorrelationTimes:

    def test_matches_1d_estimator_per_column(self):
        rng = np.random.default_rng(0)
        white = rng.normal(size=(5000, 3))
        x = np.repeat(white[::4], 4, axis=0)[:5000]   # tau ~ 4
        taus = integrated_autocorrelation_times(x)
        for j in range(3):
            assert taus[j] == pytest.approx(
                estimate_autocorrelation_time(x[:, j], quiet=True))

    def test_constant_column_gets_one(self):
        x = np.column_stack([np.zeros(100), np.random.default_rng(1).normal(size=100)])
        assert integrated_autocorrelation_times(x)[0] == 1.0

    def test_chunking_does_not_change_result(self):
        x = np.random.default_rng(2).normal(size=(500, 7))
        np.testing.assert_allclose(integrated_autocorrelation_times(x, chunk=2),
                                   integrated_autocorrelation_times(x))

    def test_rejects_1d(self):
        with pytest.raises(ValueError):
            integrated_autocorrelation_times(np.zeros(10))


# ============================================================
# confidence_halfwidth
# ============================================================

class TestConfidenceHalfwidth:

    def test_matches_student_t(self):
        from scipy.stats import t
        hw = confidence_halfwidth([1.0, 2.0], [5.0, 11.0], level=0.9)
        np.testing.assert_allclose(hw, [t.ppf(0.95, 4), 2.0 * t.ppf(0.95, 10)])

    def test_large_neff_approaches_normal(self):
        np.testing.assert_allclose(confidence_halfwidth(1.0, 1e7), 1.959964, rtol=1e-5)

    def test_no_degrees_of_freedom_gives_nan(self):
        hw = confidence_halfwidth([1.0, 1.0, 1.0], [0.5, 1.0, 3.0])
        assert np.isnan(hw[0]) and np.isnan(hw[1]) and np.isfinite(hw[2])

    def test_rejects_bad_level(self):
        with pytest.raises(ValueError):
            confidence_halfwidth(1.0, 10.0, level=95)


# ============================================================
# blocking analysis
# ============================================================

def ar1(rng, n, tau, var, n_series=1):
    """AR(1) series with integrated autocorrelation time tau and variance var.

    The variance of the mean of n samples is var * tau / n (for n >> tau).
    """
    from scipy.signal import lfilter
    phi = (tau - 1.0) / (tau + 1.0)
    burn = int(20 * tau) + 100
    e = rng.normal(scale=np.sqrt(var * (1 - phi**2)), size=(n + burn, n_series))
    return lfilter([1.0], [1.0, -phi], e, axis=0)[burn:]


class TestBlockingLevels:

    def test_doubling_block_sizes_until_min_blocks(self):
        sizes, counts, se, err = blocking_levels(np.zeros(1000), min_blocks=20)
        np.testing.assert_array_equal(sizes, [1, 2, 4, 8, 16, 32])
        np.testing.assert_array_equal(counts, 1000 // sizes)
        assert se.shape == err.shape == (6, 1)

    def test_level_matches_block_average(self):
        x = np.random.default_rng(0).normal(size=640)
        sizes, counts, se, _ = blocking_levels(x, min_blocks=20)
        for b, s in zip(sizes, se[:, 0]):
            assert s == pytest.approx(block_average(x, b)["stderr"])

    def test_stderr_error_formula(self):
        x = np.random.default_rng(1).normal(size=400)
        _, counts, se, err = blocking_levels(x)
        np.testing.assert_allclose(err[:, 0], se[:, 0] / np.sqrt(2 * (counts - 1)))

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="min_blocks"):
            blocking_levels(np.zeros(10), min_blocks=20)


class TestPlateauLevel:

    def test_white_noise_plateaus_at_first_level(self):
        x = np.random.default_rng(2).normal(size=(20000, 1))
        _, _, se, err = blocking_levels(x)
        level, ok = plateau_level(se, err)
        assert ok[0] and level[0] <= 1

    def test_runs_of_identical_values_need_blocks_as_long_as_the_runs(self):
        k = 16
        x = np.repeat(np.random.default_rng(3).normal(size=2000), k)[:, None]
        sizes, _, se, err = blocking_levels(x)
        level, ok = plateau_level(se, err)
        assert ok[0] and sizes[level[0]] >= k

    def test_unresolvable_correlation_has_no_plateau(self):
        # tau of 5000 frames, but at least 20 blocks of 20000 frames allow
        # blocks of at most 1000: the standard error keeps growing.
        x = ar1(np.random.default_rng(4), 20000, 5000.0, 1.0)
        _, _, se, err = blocking_levels(x)
        level, ok = plateau_level(se, err)
        assert not ok[0] and level[0] == len(se) - 1

    def test_constant_column_has_plateau_at_level_zero(self):
        x = np.column_stack([np.ones(500), np.random.default_rng(5).normal(size=500)])
        _, _, se, err = blocking_levels(x)
        level, ok = plateau_level(se, err)
        assert level[0] == 0 and ok[0]


class TestBlockedStderr:

    def test_given_block_size_is_plain_block_average(self):
        x = np.random.default_rng(6).normal(size=1000)
        out = blocked_stderr(x, block_size=25)
        ref = block_average(x, 25)
        assert out["stderr"] == pytest.approx(ref["stderr"])
        assert out["n_blocks"] == ref["n_blocks"] == 40 and out["block_size"] == 25

    def test_one_block_size_for_all_columns_set_by_slowest(self):
        rng = np.random.default_rng(7)
        fast = rng.normal(size=32000)
        slow = np.repeat(rng.normal(size=1000), 32)
        both = blocked_stderr(np.column_stack([fast, slow]))
        assert both["block_size"] == blocked_stderr(slow)["block_size"] >= 32
        assert both["block_size"] > blocked_stderr(fast)["block_size"]

    def test_too_few_blocks_raises(self):
        with pytest.raises(ValueError, match="min_blocks"):
            blocked_stderr(np.zeros(100), block_size=10)

    def test_tau_recovers_known_correlation_time(self):
        # AR(1) with integrated tau = 20 frames; averaged over 50 series.
        x = ar1(np.random.default_rng(20), 40000, 20.0, 1.0, n_series=50)
        taus = [blocked_stderr(x[:, j])["tau"] for j in range(50)]
        assert np.mean(taus) == pytest.approx(20.0, rel=0.1)

    def test_tau_of_white_noise_is_one(self):
        x = np.random.default_rng(21).normal(size=(40000, 50))
        taus = [blocked_stderr(x[:, j])["tau"] for j in range(50)]
        assert np.mean(taus) == pytest.approx(1.0, rel=0.1)

    def test_tau_and_n_ind_restate_the_stderr(self):
        x = ar1(np.random.default_rng(22), 10000, 10.0, 1.0)[:, 0]
        out = blocked_stderr(x)
        n = out["n_blocks"] * out["block_size"]
        s2 = x[:n].var(ddof=1)
        assert out["stderr"] == pytest.approx(np.sqrt(s2 * out["tau"] / n))
        assert out["n_ind"] == pytest.approx(n / out["tau"])

    def test_tau_of_constant_column_is_one(self):
        x = np.column_stack([np.ones(500), np.random.default_rng(23).normal(size=500)])
        assert blocked_stderr(x)["tau"][0] == 1.0

    def test_deterministic(self):
        x = ar1(np.random.default_rng(8), 10000, 20.0, 1.0, n_series=5)
        a, b = blocked_stderr(x), blocked_stderr(x.copy())
        np.testing.assert_array_equal(a["stderr"], b["stderr"])
        assert a["block_size"] == b["block_size"]


class TestCoverage:
    """99% intervals must contain the true mean (zero) about 99% of the time.

    Over 400 independent series the coverage has a binomial spread of about
    0.005 around 0.99, so >= 0.97 fails only for a real bias. (A slow
    component of a few hundred frames leaves blocks only a few correlation
    times long, which biases the error slightly low; 0.96 there.) The
    series are fixed by the seed, so the outcome is deterministic.
    """

    N, R, LEVEL = 20000, 400, 0.99

    def coverage_blocked(self, x):
        hit = []
        for j in range(x.shape[1]):
            out = blocked_stderr(x[:, j])
            hw = confidence_halfwidth(out["stderr"], out["n_blocks"], self.LEVEL)
            hit.append(abs(x[:, j].mean()) <= hw)
        return np.mean(hit)

    def coverage_autocorrelation(self, x):
        tau = integrated_autocorrelation_times(x)
        se = x.std(axis=0) * np.sqrt(tau / len(x))
        hw = confidence_halfwidth(se, len(x) / tau, self.LEVEL)
        return np.mean(np.abs(x.mean(axis=0)) <= hw)

    def test_white_noise(self):
        x = np.random.default_rng(10).normal(size=(self.N, self.R))
        assert self.coverage_blocked(x) >= 0.97

    def test_single_timescale(self):
        x = ar1(np.random.default_rng(11), self.N, 20.0, 1.0, self.R)
        assert self.coverage_blocked(x) >= 0.97

    def test_fast_plus_weak_slow_component(self):
        # The case of RDF bins: fast, large fluctuations on top of a slow,
        # small one. Blocking covers it; Sokal's windowed autocorrelation
        # time truncates the slow tail and undercovers.
        rng = np.random.default_rng(12)
        x = ar1(rng, self.N, 2.0, 1.0, self.R) + ar1(rng, self.N, 200.0, 0.05, self.R)
        assert self.coverage_blocked(x) >= 0.96
        assert self.coverage_autocorrelation(x) < 0.90
