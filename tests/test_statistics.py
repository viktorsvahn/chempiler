"""Tests for block-averaging statistics (chempiler.core.statistics)."""

import numpy as np
import pytest

from chempiler.core.statistics import split_blocks, block_average


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
