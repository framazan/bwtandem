import math
import pytest
from src.tier3 import compute_adaptive_params


class TestComputeAdaptiveParams:
    """Test adaptive parameter computation."""

    def test_balanced_1mbp(self):
        """1 Mbp sequence, 40% GC, no prior coverage."""
        params = compute_adaptive_params(
            seq_len=1_000_000, gc_content=0.4, coverage_ratio=0.0,
            min_period=100, max_period=100_000, preset="balanced"
        )
        assert 12 <= params["kmer_size"] <= 28
        assert 20 <= params["stride"] <= 300
        assert 0.15 <= params["allowed_mismatch_rate"] <= 0.20
        assert 0.02 <= params["tolerance_ratio"] <= 0.04
        assert 200 <= params["max_occurrences"] <= 1500
        assert 0.70 <= params["anchor_match_pct"] <= 0.80
        assert 20 <= params["scan_backward"] <= 80
        assert 200 <= params["scan_forward"] <= 800

    def test_micro_mode_triggers(self):
        """Sequences < 100 kbp should get micro mode floors."""
        params = compute_adaptive_params(
            seq_len=50_000, gc_content=0.5, coverage_ratio=0.0,
            min_period=100, max_period=100_000, preset="balanced"
        )
        assert params["stride"] == 20
        assert params["kmer_size"] == 12

    def test_large_chr_mode_triggers(self):
        """Sequences > 100 Mbp should get large-chr mode."""
        params = compute_adaptive_params(
            seq_len=200_000_000, gc_content=0.4, coverage_ratio=0.0,
            min_period=100, max_period=100_000, preset="balanced"
        )
        assert params["stride"] >= 150
        assert params["kmer_size"] >= 20
        assert params["max_occurrences"] <= 500

    def test_fast_increases_stride(self):
        """Fast preset should have larger stride than balanced."""
        balanced = compute_adaptive_params(
            seq_len=10_000_000, gc_content=0.4, coverage_ratio=0.0,
            min_period=100, max_period=100_000, preset="balanced"
        )
        fast = compute_adaptive_params(
            seq_len=10_000_000, gc_content=0.4, coverage_ratio=0.0,
            min_period=100, max_period=100_000, preset="fast"
        )
        assert fast["stride"] > balanced["stride"]
        assert fast["max_occurrences"] < balanced["max_occurrences"]

    def test_sensitive_decreases_stride(self):
        """Sensitive preset should have smaller stride than balanced."""
        balanced = compute_adaptive_params(
            seq_len=10_000_000, gc_content=0.4, coverage_ratio=0.0,
            min_period=100, max_period=100_000, preset="balanced"
        )
        sensitive = compute_adaptive_params(
            seq_len=10_000_000, gc_content=0.4, coverage_ratio=0.0,
            min_period=100, max_period=100_000, preset="sensitive"
        )
        assert sensitive["stride"] < balanced["stride"]
        assert sensitive["max_occurrences"] > balanced["max_occurrences"]

    def test_high_gc_increases_mismatch_rate(self):
        """Extreme GC content should increase allowed mismatch rate."""
        normal = compute_adaptive_params(
            seq_len=10_000_000, gc_content=0.5, coverage_ratio=0.0,
            min_period=100, max_period=100_000, preset="balanced"
        )
        high_gc = compute_adaptive_params(
            seq_len=10_000_000, gc_content=0.7, coverage_ratio=0.0,
            min_period=100, max_period=100_000, preset="balanced"
        )
        assert high_gc["allowed_mismatch_rate"] > normal["allowed_mismatch_rate"]

    def test_high_coverage_reduces_stride(self):
        """High prior tier coverage should reduce stride."""
        low_cov = compute_adaptive_params(
            seq_len=10_000_000, gc_content=0.4, coverage_ratio=0.1,
            min_period=100, max_period=100_000, preset="balanced"
        )
        high_cov = compute_adaptive_params(
            seq_len=10_000_000, gc_content=0.4, coverage_ratio=0.8,
            min_period=100, max_period=100_000, preset="balanced"
        )
        assert high_cov["stride"] < low_cov["stride"]

    def test_high_coverage_relaxes_anchor(self):
        """High coverage should lower anchor match threshold."""
        low_cov = compute_adaptive_params(
            seq_len=10_000_000, gc_content=0.4, coverage_ratio=0.0,
            min_period=100, max_period=100_000, preset="balanced"
        )
        high_cov = compute_adaptive_params(
            seq_len=10_000_000, gc_content=0.4, coverage_ratio=0.9,
            min_period=100, max_period=100_000, preset="balanced"
        )
        assert high_cov["anchor_match_pct"] < low_cov["anchor_match_pct"]

    def test_preset_does_not_affect_accuracy_params(self):
        """Mismatch rate, tolerance ratio, anchor pct should be preset-independent."""
        fast = compute_adaptive_params(
            seq_len=10_000_000, gc_content=0.4, coverage_ratio=0.3,
            min_period=100, max_period=100_000, preset="fast"
        )
        sensitive = compute_adaptive_params(
            seq_len=10_000_000, gc_content=0.4, coverage_ratio=0.3,
            min_period=100, max_period=100_000, preset="sensitive"
        )
        assert fast["allowed_mismatch_rate"] == sensitive["allowed_mismatch_rate"]
        assert fast["tolerance_ratio"] == sensitive["tolerance_ratio"]
        assert fast["anchor_match_pct"] == sensitive["anchor_match_pct"]

    def test_returns_all_keys(self):
        """Function must return all 8 expected keys."""
        params = compute_adaptive_params(
            seq_len=5_000_000, gc_content=0.45, coverage_ratio=0.2,
            min_period=100, max_period=100_000, preset="balanced"
        )
        expected_keys = {
            "kmer_size", "stride", "allowed_mismatch_rate", "tolerance_ratio",
            "max_occurrences", "anchor_match_pct", "scan_backward", "scan_forward"
        }
        assert set(params.keys()) == expected_keys
