"""Optional accelerators backed by the Cython extension."""
from __future__ import annotations  # Allow forward references in type hints for Python < 3.9

from typing import Optional, Tuple  # Import Optional, Tuple for type hints

import numpy as np  # Import NumPy for numerical array processing

try:
    from . import _accelerators as _native  # type: ignore  # Attempt to import compiled Cython extension module
except Exception:  # pragma: no cover - fallback to pyximport
    # If Cython .so file is not available, attempt runtime compilation via pyximport
    _native = None  # Set to None due to native module initialization failure
    try:
        import pyximport  # type: ignore  # Import runtime Cython compiler

        pyximport.install(  # type: ignore[attr-defined]
            language_level=3,           # Use Python 3 syntax
            inplace=True,               # Build in the source directory
            build_in_temp=False,        # Do not use a temporary directory
            setup_args={"include_dirs": np.get_include()},  # Include NumPy header paths
        )
        from . import _accelerators as _native  # type: ignore  # pylint: disable=import-error  # Re-import after compilation via pyximport
    except Exception as e:
        print(f"DEBUG: importing _accelerators failed: {e}")  # Debug output showing import failure cause
        _native = None  # Set to None after all attempts fail; use pure-Python fallback


AcceleratorResult = Optional[Tuple[int, int, int, int, int]]  # Type alias for accelerator function return (start, end, copies, etc.)


if _native is not None:
    # Direct alias for maximum performance
    # Set direct aliases to Cython native functions for maximum performance
    hamming_distance = _native.hamming_distance        # Compute Hamming distance between two arrays
    extend_with_mismatches = _native.extend_with_mismatches  # Extend repeat boundaries allowing mismatches
    pack_sequence = _native.pack_sequence              # Compress DNA sequence into 2-bit packed format

    # Wrap scan_unit_repeats to ensure it appears in profiler output
    # (The overhead is negligible as it's called once per unit_len)
    # Wrap in a function so it appears in profiler output (called once per unit_len, so overhead is negligible)
    def scan_unit_repeats(
        text_arr: np.ndarray,               # Sequence byte array to analyze
        n: int,                             # Effective sequence length (excluding sentinel)
        unit_len: int,                      # Repeat unit (motif) length to search for
        min_copies: int,                    # Minimum number of copies to qualify as a repeat
        max_mismatch: int,                  # Maximum number of mismatches allowed
        packed_arr: Optional[np.ndarray] = None  # Pre-packed sequence array (optional)
    ) -> list:
        return _native.scan_unit_repeats(text_arr, n, unit_len, min_copies, max_mismatch, packed_arr)
        # Call Cython scan_unit_repeats to return list of repeat sequence positions

    def scan_simple_repeats(
        text_arr: np.ndarray,               # Sequence byte array to analyze
        tier1_mask: np.ndarray,             # Mask array indicating positions already found by Tier 1
        n: int,                             # Effective sequence length
        min_p: int,                         # Minimum period length to search
        max_p: int,                         # Maximum period length to search
        period_step: int,                   # Period search interval (step size)
        position_step: int,                 # Position search interval (step size)
        allowed_mismatch_rate: float        # Allowed mismatch rate
    ) -> list:
        return _native.scan_simple_repeats(
            text_arr, tier1_mask, n, min_p, max_p, period_step, position_step, allowed_mismatch_rate
        )  # Call Cython sliding-window repeat scan

    def find_periodic_patterns(
        positions: np.ndarray,              # K-mer occurrence position array from FM-index
        min_period: int,                    # Minimum period to search
        max_period: int,                    # Maximum period to search
        min_copies: int,                    # Minimum number of copies
        tolerance_ratio: float = 0.01       # Arithmetic progression tolerance ratio
    ) -> list:
        return _native.find_periodic_patterns(positions, min_period, max_period, min_copies, tolerance_ratio)
        # Cython implementation: detect arithmetic progressions (periodic patterns) in position array

    def find_periodic_runs(
        positions: np.ndarray,              # K-mer occurrence position array from FM-index
        min_period: int,                    # Minimum period to search
        max_period: int,                    # Maximum period to search
        min_copies: int,                    # Minimum number of copies
        tolerance_ratio: float = 0.01       # Tolerance ratio
    ) -> list:
        return _native.find_periodic_runs(positions, min_period, max_period, min_copies, tolerance_ratio)
        # Cython implementation: find consecutive periodic runs

    def align_unit_to_window(
        motif: bytes,                       # Reference motif byte string for alignment
        window: bytes,                      # Sequence window byte string to compare
        max_indel: int,                     # Maximum number of insertions/deletions allowed
        mismatch_tolerance: int             # Maximum number of mismatches allowed
    ) -> Optional[Tuple]:
        return _native.align_unit_to_window(motif, window, max_indel, mismatch_tolerance)
        # Cython implementation: perform alignment between motif and window, return result

    def lcp_tandem_candidates(
        sa: np.ndarray,                     # Suffix array
        lcp: np.ndarray,                    # LCP (longest common prefix) array
        n: int,                             # Effective sequence length
        min_period: int,                    # Minimum period to search
        max_period: int,                    # Maximum period to search
        min_lcp_threshold: int = 10         # Minimum LCP threshold (for noise filtering)
    ) -> list:
        return _native.lcp_tandem_candidates(sa, lcp, n, min_period, max_period, min_lcp_threshold)
        # Cython implementation: return tandem repeat candidates (period, start_pos) from LCP array

    def find_tandem_runs(
        positions: np.ndarray,              # K-mer occurrence position array
        period: int,                        # Period length to search for
        min_copies: int                     # Minimum number of copies
    ) -> list:
        return _native.find_tandem_runs(positions, period, min_copies)
        # Cython implementation: find tandem runs of specified period in position array

    def anchor_scan_boundaries(
        text_arr: np.ndarray,               # Sequence byte array
        seed_pos: int,                      # Seed (anchor) position
        period: int,                        # Repeat period length
        n: int,                             # Effective sequence length
        match_threshold: float,             # Copy match ratio threshold
        max_backward_periods: int,          # Maximum number of periods to scan backward
        max_forward_periods: int,           # Maximum number of periods to scan forward
    ) -> Tuple[int, int]:
        return _native.anchor_scan_boundaries(
            text_arr, seed_pos, period, n, match_threshold,
            max_backward_periods, max_forward_periods
        )  # Cython implementation: scan repeat array boundaries bidirectionally from anchor, return (start, end)
else:
    # Define pure-Python fallback functions when Cython module is unavailable
    def hamming_distance(arr1: np.ndarray, arr2: np.ndarray) -> Optional[int]:
        return None  # Fallback: cannot compute Hamming distance, return None

    def extend_with_mismatches(
        s_arr: np.ndarray,                  # Sequence array
        start_pos: int,                     # Search start position
        period: int,                        # Repeat period
        n: int,                             # Effective sequence length
        allowed_mismatch_rate: float,       # Allowed mismatch rate
    ) -> AcceleratorResult:
        return None  # Fallback: cannot extend with mismatches, return None

    def pack_sequence(text_arr: np.ndarray) -> np.ndarray:
        return np.array([], dtype=np.uint8)  # Fallback: return empty array (packing unavailable)

    def scan_unit_repeats(
        text_arr: np.ndarray,               # Sequence array
        n: int,                             # Effective sequence length
        unit_len: int,                      # Repeat unit length
        min_copies: int,                    # Minimum number of copies
        max_mismatch: int,                  # Maximum mismatches allowed
        packed_arr: Optional[np.ndarray] = None  # Pre-packed array (optional)
    ) -> list:
        return []  # Fallback: return empty result (large-scale scan unavailable without Cython)

    def scan_simple_repeats(
        text_arr: np.ndarray,               # Sequence array
        tier1_mask: np.ndarray,             # Tier 1 mask array
        n: int,                             # Effective sequence length
        min_p: int,                         # Minimum period
        max_p: int,                         # Maximum period
        period_step: int,                   # Period step
        position_step: int,                 # Position step
        allowed_mismatch_rate: float        # Allowed mismatch rate
    ) -> list:
        return []  # Fallback: return empty result

    def find_periodic_patterns(
        positions: np.ndarray,              # Position array
        min_period: int,                    # Minimum period
        max_period: int,                    # Maximum period
        min_copies: int,                    # Minimum number of copies
        tolerance_ratio: float = 0.01       # Tolerance ratio
    ) -> list:
        return []  # Fallback: return empty result

    def align_unit_to_window(
        motif: bytes,                       # Motif byte string
        window: bytes,                      # Window byte string
        max_indel: int,                     # Maximum insertions/deletions
        mismatch_tolerance: int             # Maximum mismatches
    ) -> Optional[Tuple]:
        return None  # Fallback: cannot perform alignment, return None

    def find_periodic_runs(
        positions: np.ndarray,              # Position array
        min_period: int,                    # Minimum period
        max_period: int,                    # Maximum period
        min_copies: int,                    # Minimum number of copies
        tolerance_ratio: float = 0.01       # Tolerance ratio
    ) -> list:
        return []  # Fallback: return empty result

    def lcp_tandem_candidates(
        sa: np.ndarray,                     # Suffix array
        lcp: np.ndarray,                    # LCP array
        n: int,                             # Effective sequence length
        min_period: int,                    # Minimum period
        max_period: int,                    # Maximum period
        min_lcp_threshold: int = 10         # Minimum LCP threshold
    ) -> list:
        """Pure-Python fallback for LCP tandem candidate detection."""
        results = []  # Tandem repeat candidate result list
        sa_len = len(sa)    # Suffix array length
        lcp_len = len(lcp)  # LCP array length
        limit = min(sa_len, lcp_len)  # Limit iteration range to the shorter of the two arrays
        for i in range(1, limit):
            L = int(lcp[i])  # LCP value at current position
            if L < min_lcp_threshold:
                continue  # Skip if LCP value is below threshold (considered noise)
            pos_a = int(sa[i - 1])  # Start position of the previous suffix
            pos_b = int(sa[i])      # Start position of the current suffix
            if pos_a >= n or pos_b >= n:
                continue  # Skip positions beyond the sentinel '$'
            diff = abs(pos_b - pos_a)  # Position difference between two suffixes = potential period length
            if diff < min_period or diff > max_period:
                continue  # Skip if outside the period range
            start = min(pos_a, pos_b)  # The smaller of the two positions is the repeat start
            results.append((diff, start))  # Add (period, start_position) tuple to results
        return results  # Return tandem repeat candidate list

    def find_tandem_runs(
        positions: np.ndarray,              # K-mer occurrence position array
        period: int,                        # Period to search for
        min_copies: int                     # Minimum number of copies
    ) -> list:
        """Pure-Python fallback for tandem run detection."""
        n_pos = len(positions)  # Number of elements in the position array
        if n_pos < min_copies:
            return []  # Return empty result if position count is less than minimum copies
        results = []  # Tandem run result list
        run_start = int(positions[0])  # Start position of the current run
        expected_next = run_start + period  # Expected value of the next position (arithmetic progression)
        count = 1  # Length counter for the current run
        for i in range(1, n_pos):
            if int(positions[i]) == expected_next:
                # Matches expected position, continue the run
                count += 1  # Increment copy count
                expected_next = int(positions[i]) + period  # Update next expected position
            else:
                # Continuity broken: save previous run and start a new one
                if count >= min_copies:
                    results.append((run_start, expected_next))  # Add to results if enough copies
                run_start = int(positions[i])  # Start position of the new run
                expected_next = run_start + period  # Next expected position for the new run
                count = 1  # Reset counter
        if count >= min_copies:
            results.append((run_start, expected_next))  # Handle the last run
        return results  # Return tandem run list

    def anchor_scan_boundaries(
        text_arr: np.ndarray,               # Sequence byte array
        seed_pos: int,                      # Seed (anchor) position
        period: int,                        # Repeat period length
        n: int,                             # Effective sequence length
        match_threshold: float,             # Copy match ratio threshold
        max_backward_periods: int,          # Maximum number of periods to scan backward
        max_forward_periods: int,           # Maximum number of periods to scan forward
    ) -> Tuple[int, int]:
        """Pure-Python fallback for anchor-based boundary scanning."""
        if seed_pos + period > n:
            return (seed_pos, seed_pos + period)  # Return minimal range if seed is at the end of the sequence

        motif_arr = text_arr[seed_pos:seed_pos + period]  # Extract motif array at seed position
        true_start = seed_pos        # Actual start of the repeat array (initial: seed position)
        true_end = seed_pos + period  # Actual end of the repeat array (initial: seed + period)

        # Scan backward (5' direction) from seed to find repeat boundary
        scan_start = max(0, seed_pos - period * max_backward_periods)  # Backward scan limit position
        pos = seed_pos - period  # Backward scan start position
        while pos >= scan_start:
            window = text_arr[pos:pos + period]  # Extract window array at current position
            if window.size == period:
                # Count bases matching between window and motif
                matches = int(np.sum(window == motif_arr))
                if matches / period >= match_threshold:
                    # Match ratio meets threshold: extend start position backward
                    true_start = pos
                    pos -= period  # Move one more period backward
                else:
                    break  # Match ratio below threshold: stop backward scan
            else:
                break  # Window size mismatch: stop scan

        # Scan forward (3' direction) from seed to find repeat boundary
        scan_end = min(n, seed_pos + period * max_forward_periods)  # Forward scan limit position
        pos = seed_pos + period  # Forward scan start position
        while pos + period <= scan_end:
            window = text_arr[pos:pos + period]  # Extract window array at current position
            if window.size == period:
                # Count bases matching between window and motif
                matches = int(np.sum(window == motif_arr))
                if matches / period >= match_threshold:
                    # Match ratio meets threshold: extend end position forward
                    true_end = pos + period
                    pos += period  # Move one more period forward
                else:
                    break  # Match ratio below threshold: stop forward scan
            else:
                break  # Window size mismatch: stop scan

        return (true_start, true_end)  # Return (start, end) coordinates of the detected repeat array
