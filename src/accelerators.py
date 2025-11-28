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
