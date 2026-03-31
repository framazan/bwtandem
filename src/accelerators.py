"""Optional accelerators backed by the Cython extension."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    from . import _accelerators as _native  # type: ignore
except Exception:  # pragma: no cover - fallback to pyximport
    _native = None
    try:
        import pyximport  # type: ignore

        pyximport.install(  # type: ignore[attr-defined]
            language_level=3,
            inplace=True,
            build_in_temp=False,
            setup_args={"include_dirs": np.get_include()},
        )
        from . import _accelerators as _native  # type: ignore  # pylint: disable=import-error
    except Exception as e:
        print(f"DEBUG: importing _accelerators failed: {e}")
        _native = None


AcceleratorResult = Optional[Tuple[int, int, int, int, int]]


if _native is not None:
    # Direct alias for maximum performance
    hamming_distance = _native.hamming_distance
    extend_with_mismatches = _native.extend_with_mismatches
    pack_sequence = _native.pack_sequence
    
    # Wrap scan_unit_repeats to ensure it appears in profiler output
    # (The overhead is negligible as it's called once per unit_len)
    def scan_unit_repeats(
        text_arr: np.ndarray,
        n: int,
        unit_len: int,
        min_copies: int,
        max_mismatch: int,
        packed_arr: Optional[np.ndarray] = None
    ) -> list:
        return _native.scan_unit_repeats(text_arr, n, unit_len, min_copies, max_mismatch, packed_arr)

    def scan_simple_repeats(
        text_arr: np.ndarray,
        tier1_mask: np.ndarray,
        n: int,
        min_p: int,
        max_p: int,
        period_step: int,
        position_step: int,
        allowed_mismatch_rate: float
    ) -> list:
        return _native.scan_simple_repeats(
            text_arr, tier1_mask, n, min_p, max_p, period_step, position_step, allowed_mismatch_rate
        )

    def find_periodic_patterns(
        positions: np.ndarray,
        min_period: int,
        max_period: int,
        min_copies: int,
        tolerance_ratio: float = 0.01
    ) -> list:
        return _native.find_periodic_patterns(positions, min_period, max_period, min_copies, tolerance_ratio)

    def find_periodic_runs(
        positions: np.ndarray,
        min_period: int,
        max_period: int,
        min_copies: int,
        tolerance_ratio: float = 0.01
    ) -> list:
        return _native.find_periodic_runs(positions, min_period, max_period, min_copies, tolerance_ratio)

    def align_unit_to_window(
        motif: bytes,
        window: bytes,
        max_indel: int,
        mismatch_tolerance: int
    ) -> Optional[Tuple]:
        return _native.align_unit_to_window(motif, window, max_indel, mismatch_tolerance)

    def lcp_tandem_candidates(
        sa: np.ndarray,
        lcp: np.ndarray,
        n: int,
        min_period: int,
        max_period: int,
        min_lcp_threshold: int = 10
    ) -> list:
        return _native.lcp_tandem_candidates(sa, lcp, n, min_period, max_period, min_lcp_threshold)

    def find_tandem_runs(
        positions: np.ndarray,
        period: int,
        min_copies: int
    ) -> list:
        return _native.find_tandem_runs(positions, period, min_copies)

    def anchor_scan_boundaries(
        text_arr: np.ndarray,
        seed_pos: int,
        period: int,
        n: int,
        match_threshold: float,
        max_backward_periods: int,
        max_forward_periods: int,
    ) -> Tuple[int, int]:
        return _native.anchor_scan_boundaries(
            text_arr, seed_pos, period, n, match_threshold,
            max_backward_periods, max_forward_periods
        )
else:
    def hamming_distance(arr1: np.ndarray, arr2: np.ndarray) -> Optional[int]:
        return None

    def extend_with_mismatches(
        s_arr: np.ndarray,
        start_pos: int,
        period: int,
        n: int,
        allowed_mismatch_rate: float,
    ) -> AcceleratorResult:
        return None
        
    def pack_sequence(text_arr: np.ndarray) -> np.ndarray:
        return np.array([], dtype=np.uint8)

    def scan_unit_repeats(
        text_arr: np.ndarray,
        n: int,
        unit_len: int,
        min_copies: int,
        max_mismatch: int,
        packed_arr: Optional[np.ndarray] = None
    ) -> list:
        return []

    def scan_simple_repeats(
        text_arr: np.ndarray,
        tier1_mask: np.ndarray,
        n: int,
        min_p: int,
        max_p: int,
        period_step: int,
        position_step: int,
        allowed_mismatch_rate: float
    ) -> list:
        return []

    def find_periodic_patterns(
        positions: np.ndarray,
        min_period: int,
        max_period: int,
        min_copies: int,
        tolerance_ratio: float = 0.01
    ) -> list:
        return []

    def align_unit_to_window(
        motif: bytes,
        window: bytes,
        max_indel: int,
        mismatch_tolerance: int
    ) -> Optional[Tuple]:
        return None

    def find_periodic_runs(
        positions: np.ndarray,
        min_period: int,
        max_period: int,
        min_copies: int,
        tolerance_ratio: float = 0.01
    ) -> list:
        return []

    def lcp_tandem_candidates(
        sa: np.ndarray,
        lcp: np.ndarray,
        n: int,
        min_period: int,
        max_period: int,
        min_lcp_threshold: int = 10
    ) -> list:
        """Pure-Python fallback for LCP tandem candidate detection."""
        results = []
        sa_len = len(sa)
        lcp_len = len(lcp)
        limit = min(sa_len, lcp_len)
        for i in range(1, limit):
            L = int(lcp[i])
            if L < min_lcp_threshold:
                continue
            pos_a = int(sa[i - 1])
            pos_b = int(sa[i])
            if pos_a >= n or pos_b >= n:
                continue
            diff = abs(pos_b - pos_a)
            if diff < min_period or diff > max_period:
                continue
            start = min(pos_a, pos_b)
            results.append((diff, start))
        return results

    def find_tandem_runs(
        positions: np.ndarray,
        period: int,
        min_copies: int
    ) -> list:
        """Pure-Python fallback for tandem run detection."""
        n_pos = len(positions)
        if n_pos < min_copies:
            return []
        results = []
        run_start = int(positions[0])
        expected_next = run_start + period
        count = 1
        for i in range(1, n_pos):
            if int(positions[i]) == expected_next:
                count += 1
                expected_next = int(positions[i]) + period
            else:
                if count >= min_copies:
                    results.append((run_start, expected_next))
                run_start = int(positions[i])
                expected_next = run_start + period
                count = 1
        if count >= min_copies:
            results.append((run_start, expected_next))
        return results

    def anchor_scan_boundaries(
        text_arr: np.ndarray,
        seed_pos: int,
        period: int,
        n: int,
        match_threshold: float,
        max_backward_periods: int,
        max_forward_periods: int,
    ) -> Tuple[int, int]:
        """Pure-Python fallback for anchor-based boundary scanning."""
        if seed_pos + period > n:
            return (seed_pos, seed_pos + period)

        motif_arr = text_arr[seed_pos:seed_pos + period]
        true_start = seed_pos
        true_end = seed_pos + period

        # Scan backward
        scan_start = max(0, seed_pos - period * max_backward_periods)
        pos = seed_pos - period
        while pos >= scan_start:
            window = text_arr[pos:pos + period]
            if window.size == period:
                matches = int(np.sum(window == motif_arr))
                if matches / period >= match_threshold:
                    true_start = pos
                    pos -= period
                else:
                    break
            else:
                break

        # Scan forward
        scan_end = min(n, seed_pos + period * max_forward_periods)
        pos = seed_pos + period
        while pos + period <= scan_end:
            window = text_arr[pos:pos + period]
            if window.size == period:
                matches = int(np.sum(window == motif_arr))
                if matches / period >= match_threshold:
                    true_end = pos + period
                    pos += period
                else:
                    break
            else:
                break

        return (true_start, true_end)
