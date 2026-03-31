import numpy as np
import pytest
from src.accelerators import anchor_scan_boundaries


class TestAnchorScanBoundaries:
    """Test anchor_scan_boundaries (pure-Python fallback)."""

    def _make_repeat(self, motif: str, copies: int, prefix: str = "", suffix: str = "") -> np.ndarray:
        seq = prefix + motif * copies + suffix
        return np.array(list(seq.encode('ascii')), dtype=np.uint8)

    def test_perfect_repeat(self):
        """Perfect tandem repeat should extend fully."""
        arr = self._make_repeat("ACGT", 20)
        # Seed at copy 10
        seed = 10 * 4
        start, end = anchor_scan_boundaries(arr, seed, 4, len(arr), 0.75, 50, 50)
        assert start == 0
        assert end == 80

    def test_flanked_repeat(self):
        """Repeat with non-matching flanks should stop at boundaries."""
        arr = self._make_repeat("ACGT", 10, prefix="TTTTTTTT", suffix="GGGGGGGG")
        n = len(arr)
        seed = 8 + 4 * 4  # middle of repeat region
        start, end = anchor_scan_boundaries(arr, seed, 4, n, 0.75, 50, 50)
        assert start == 8  # should stop at flank
        assert end == 8 + 40  # should stop before suffix

    def test_imperfect_repeat_above_threshold(self):
        """Repeat with some mismatches but above threshold should extend."""
        motif = "ACGTACGT"  # 8bp
        arr = self._make_repeat(motif, 10)
        # Introduce 1 mismatch per copy (12.5% mismatch, 87.5% match > 75%)
        for i in range(10):
            arr[i * 8 + 7] = ord('T')
        seed = 4 * 8
        start, end = anchor_scan_boundaries(arr, seed, 8, len(arr), 0.75, 50, 50)
        # Should still extend through all copies (87.5% match)
        assert start == 0
        assert end == 80

    def test_single_copy_no_extension(self):
        """Single copy surrounded by random should not extend."""
        motif = b"ACGTACGT"
        random_seq = b"TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"
        arr = np.frombuffer(random_seq + motif + random_seq, dtype=np.uint8)
        seed = 32
        start, end = anchor_scan_boundaries(arr, seed, 8, len(arr), 0.75, 50, 50)
        assert start == seed
        assert end == seed + 8

    def test_returns_tuple(self):
        arr = self._make_repeat("AC", 5)
        result = anchor_scan_boundaries(arr, 4, 2, len(arr), 0.75, 10, 10)
        assert isinstance(result, tuple)
        assert len(result) == 2
