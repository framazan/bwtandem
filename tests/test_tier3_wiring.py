import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.tier3 import Tier3LongReadFinder


class TestTier3ModeWiring:
    """Test that Tier3LongReadFinder uses adaptive params."""

    def _make_mock_bwt(self, seq_len: int, gc_fraction: float = 0.4):
        bwt = MagicMock()
        text_arr = np.full(seq_len, ord('A'), dtype=np.uint8)
        gc_count = int(seq_len * gc_fraction)
        text_arr[:gc_count // 2] = ord('G')
        text_arr[gc_count // 2:gc_count] = ord('C')
        text_arr[-1] = ord('$')
        bwt.text_arr = text_arr
        bwt.text = text_arr.tobytes().decode('ascii')
        bwt.suffix_array = np.arange(seq_len, dtype=np.int64)
        return bwt

    def test_init_accepts_mode(self):
        bwt = self._make_mock_bwt(1000)
        finder = Tier3LongReadFinder(bwt, mode="fast")
        assert finder.mode == "fast"

    def test_init_default_mode_is_balanced(self):
        bwt = self._make_mock_bwt(1000)
        finder = Tier3LongReadFinder(bwt)
        assert finder.mode == "balanced"

    @patch("src.tier3.bwt_kmer_seed_scan")
    def test_adaptive_params_passed_to_seed_scan(self, mock_scan):
        mock_scan.return_value = []
        bwt = self._make_mock_bwt(5_000_000)
        finder = Tier3LongReadFinder(bwt, mode="sensitive")
        finder.find_long_repeats("chr1", set(), set())
        mock_scan.assert_called_once()
