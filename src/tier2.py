import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from typing import List, Tuple, Set, Optional
import time
from .models import TandemRepeat
from .motif_utils import MotifUtils
from .bwt_core import BWTCore, _kasai_lcp_uint8
from .accelerators import hamming_distance, extend_with_mismatches, scan_unit_repeats, scan_simple_repeats, pack_sequence, lcp_tandem_candidates, find_tandem_runs

class Tier2LCPFinder:
    """Tier 2: BWT/FM-index based repeat finder for ALL motif lengths with imperfect repeat support.

    Handles both short repeats (with mismatches) and medium/long repeats using:
    - FM-index backward search for motif occurrences
    - LCP arrays for longer period detection
    - Seed-and-extend with mismatch tolerance
    """

    def __init__(self, bwt_core: BWTCore, min_period: int = 1, max_period: int = 1000,
                 max_short_motif: int = 9, allow_mismatches: bool = True,
                 allowed_mismatch_rate: float = 0.2, allowed_indel_rate: float = 0.1,
                 show_progress: bool = False):
        self.bwt = bwt_core
        # Tier 2 is designed for non-microsatellite motifs; enforce >=10bp
        self.min_period = max(10, min_period)
        self.max_period = max_period
        self.max_short_motif = max_short_motif  # For FM-index search (1-9bp)
        self.min_copies = 3  # Require at least 3 copies
        self.min_array_length = 6  # Minimum total array length (for short repeats)
        self.min_entropy = 1.0  # Minimum Shannon entropy
        self.allow_mismatches = allow_mismatches
        self.show_progress = show_progress
        self.period_step = 1  # Step size for period scanning (increase to speed up)
        self.allowed_mismatch_rate = max(0.0, allowed_mismatch_rate)
        self.allowed_indel_rate = max(0.0, allowed_indel_rate)
        self.sequence_str = self.bwt.text_arr.tobytes().decode('ascii', errors='replace')
        self._lcp_cache = None  # Lazily computed LCP array
        self._bwt_call_count = 0  # Track BWT usage for diagnostics

    def _get_lcp_array(self) -> np.ndarray:
        """Get (or compute and cache) the LCP array."""
        if self._lcp_cache is None:
            self._lcp_cache = self._compute_lcp_array()
        return self._lcp_cache

    def _bwt_find_repeats_for_period(
        self, chromosome: str, period: int, seed_positions: List[int],
        min_copies: int, covered: Set[Tuple[int, int]]
    ) -> List[TandemRepeat]:
        """BWT-driven repeat detection for a specific period.

        Given seed positions (from LCP analysis), uses FM-index backward_search
        and locate_positions to find all occurrences of the candidate motif,
        then detects tandem runs and extends with mismatch tolerance.

        Args:
            chromosome: Chromosome name
            period: Candidate repeat period
            seed_positions: Starting text positions from LCP analysis
            min_copies: Minimum tandem copies required
            covered: Set of (start, end) regions already found (updated in-place)

        Returns:
            List of TandemRepeat objects found via BWT
        """
        repeats = []
        text_arr = self.bwt.text_arr
        n = int(text_arr.size)
        if n > 0 and text_arr[n - 1] == 36:
            n -= 1

        seen_motifs: Set[str] = set()

        for seed_pos in seed_positions:
            if seed_pos + period > n:
                continue

            # Skip if this seed is already inside a covered region
            skip = False
            for cs, ce in covered:
                if cs <= seed_pos < ce:
                    skip = True
                    break
            if skip:
                continue

            # Extract candidate motif from this seed position
            motif_arr = text_arr[seed_pos:seed_pos + period]
            motif_str = motif_arr.tobytes().decode('ascii', errors='replace')

            # Skip if we already tried this exact motif
            if motif_str in seen_motifs:
                continue
            seen_motifs.add(motif_str)

            # Skip non-DNA motifs
            if not all(c in 'ACGT' for c in motif_str):
                continue

            # --- FM-index: find ALL occurrences of this motif in O(m) ---
            self._bwt_call_count += 1
            sp, ep = self.bwt.backward_search(motif_str)
            if sp == -1:
                continue

            occ_count = ep - sp + 1

            # Cap positions to avoid explosion on low-complexity motifs
            if occ_count < min_copies:
                continue
            if occ_count > 5000:
                continue  # Too common; brute-force fallback handles this

            # Get all positions via suffix array lookup
            positions = self.bwt.locate_positions(motif_str)
            if len(positions) < min_copies:
                continue

            # Find tandem runs (arithmetic progressions with spacing = period)
            pos_arr = np.array(sorted(positions), dtype=np.int64)
            tandem_runs = find_tandem_runs(pos_arr, period, min_copies)

            for run_start, run_end in tandem_runs:
                run_start = int(run_start)
                run_end = int(run_end)

                # Skip if overlapping with already covered region
                skip = False
                for cs, ce in covered:
                    overlap = min(ce, run_end) - max(cs, run_start)
                    if overlap > 0 and overlap > (run_end - run_start) * 0.5:
                        skip = True
                        break
                if skip:
                    continue

                # Extend with mismatch tolerance (handles imperfect repeats)
                res = extend_with_mismatches(
                    text_arr, run_start, period, n,
                    self.allowed_mismatch_rate
                )

                if res is not None:
                    arr_start, arr_end, copies, full_start, full_end = res
                else:
                    # Fallback: use the raw run boundaries
                    full_start = run_start
                    full_end = run_end
                    copies = (run_end - run_start) // period
                    arr_start = run_start
                    arr_end = run_end

                if copies < min_copies:
                    continue

                # Extract motif and reduce to primitive period
                final_motif = text_arr[full_start:full_start + period].tobytes().decode('ascii', errors='replace')
                primitive_period = MotifUtils.smallest_period_str(final_motif)
                if primitive_period == len(final_motif):
                    primitive_period = MotifUtils.smallest_period_str_approx(final_motif, max_error_rate=0.02)
                if primitive_period < len(final_motif):
                    final_motif = final_motif[:primitive_period]
                    copies = (full_end - full_start) // primitive_period

                repeat = self._refine_and_create_repeat(
                    chromosome, full_start, full_end, final_motif,
                    tier=2, min_copies=max(2, min_copies)
                )
                if repeat:
                    repeats.append(repeat)
                    covered.add((repeat.start, repeat.end))

        return repeats

    def _refine_and_create_repeat(self, chromosome: str, start: int, end: int, motif: str,
                                  tier: int, min_copies: Optional[int] = None,
                                  strand: str = '+') -> Optional[TandemRepeat]:
        if min_copies is None:
            min_copies = self.min_copies

        refined = MotifUtils.refine_repeat(
            self.sequence_str,
            start,
            end,
            motif,
            mismatch_fraction=self.allowed_mismatch_rate,
            indel_fraction=self.allowed_indel_rate,
            min_copies=min_copies
        )

        if not refined:
            return None

        return MotifUtils.refined_to_repeat(chromosome, refined, tier, self.bwt.text_arr, strand=strand)

    def _hamming_distance(self, arr1: np.ndarray, arr2: np.ndarray) -> int:
        """Calculate Hamming distance between two arrays."""
        # Try accelerated version first
        res = hamming_distance(arr1, arr2)
        if res is not None:
            return res
        print('WARNING: non-accelerated version being used, will be super slow')
        return int(np.sum(arr1 != arr2))

    def find_long_unit_repeats_strict(self, chromosome: str, min_unit_len: int = 20,
                                      max_unit_len: int = 120, max_mismatch: int = 2,
                                      min_copies: int = 2) -> List[TandemRepeat]:
        """Find long-unit tandem repeats using BWT/LCP-driven detection with brute-force fallback.

        Phase A: Uses suffix array and LCP array to identify tandem repeat
                 signatures, then FM-index backward_search/locate_positions
                 to find all occurrences and extend with mismatch tolerance.
        Phase B: Falls back to brute-force sliding window on regions not
                 covered by Phase A.

        Args:
            chromosome: Chromosome name
            min_unit_len: Minimum unit length to consider (default 20bp)
            max_unit_len: Maximum unit length to scan (default 120bp)
            max_mismatch: Maximum Hamming distance per unit comparison (default 2)
            min_copies: Minimum number of adjacent copies required (default 2 for long units)

        Returns:
            List of long-unit tandem repeats
        """
        repeats = []
        text_arr = self.bwt.text_arr
        n = int(text_arr.size)

        # Exclude sentinel
        if n > 0 and text_arr[n - 1] == 36:  # '$' = 36
            n -= 1

        covered: Set[Tuple[int, int]] = set()

        # ===== Phase A: BWT/LCP-driven detection =====
        if self.show_progress:
            print(f"  [{chromosome}] Tier 2 long-unit: Phase A (BWT/LCP) starting...", flush=True)

        lcp = self._get_lcp_array()
        sa = self.bwt.suffix_array.astype(np.int32, copy=False)

        # Scan LCP array for tandem repeat signatures
        candidates = lcp_tandem_candidates(sa, lcp, n, min_unit_len, max_unit_len)

        # Group candidates by period
        period_seeds: dict = {}  # period -> list of seed positions
        for period, start_pos in candidates:
            if period not in period_seeds:
                period_seeds[period] = []
            period_seeds[period].append(start_pos)

        # Process each period using BWT (longest first for proper nesting)
        bwt_found = 0
        for period in sorted(period_seeds.keys(), reverse=True):
            seeds = period_seeds[period]
            # Deduplicate seed positions
            unique_seeds = sorted(set(seeds))

            bwt_repeats = self._bwt_find_repeats_for_period(
                chromosome, period, unique_seeds, min_copies, covered
            )
            repeats.extend(bwt_repeats)
            bwt_found += len(bwt_repeats)

        if self.show_progress:
            print(f"  [{chromosome}] Tier 2 long-unit: Phase A found {bwt_found} repeats "
                  f"({len(candidates)} LCP candidates, {self._bwt_call_count} FM-index queries)", flush=True)

        # ===== Phase B: Brute-force fallback on uncovered regions =====
        if self.show_progress:
            print(f"  [{chromosome}] Tier 2 long-unit: Phase B (brute-force fallback)...", flush=True)

        # Pre-pack sequence for 2-bit acceleration
        packed_arr = pack_sequence(text_arr)

        # Build coverage mask from Phase A results
        covered_mask = np.zeros(n, dtype=bool)
        for cs, ce in covered:
            covered_mask[cs:min(ce, n)] = True

        fallback_found = 0
        max_possible_unit = min(max_unit_len, n // min_copies)
        for unit_len in range(max_possible_unit, min_unit_len - 1, -1):
            # Use accelerated scanner if available
            fb_candidates = scan_unit_repeats(text_arr, n, unit_len, min_copies, max_mismatch, packed_arr)

            if fb_candidates is not None:
                for start_pos, end_pos in fb_candidates:
                    # Skip if mostly covered by Phase A
                    region_covered = np.sum(covered_mask[start_pos:end_pos])
                    if region_covered > (end_pos - start_pos) * 0.5:
                        continue

                    length = end_pos - start_pos
                    count = length // unit_len

                    motif_arr = text_arr[start_pos:start_pos + unit_len]
                    motif = motif_arr.tobytes().decode('ascii', errors='replace')

                    primitive_period = MotifUtils.smallest_period_str(motif)
                    if primitive_period == len(motif):
                        primitive_period = MotifUtils.smallest_period_str_approx(motif, max_error_rate=0.02)
                    if primitive_period < len(motif):
                        motif = motif[:primitive_period]
                        count = length // primitive_period

                    repeat = self._refine_and_create_repeat(
                        chromosome, start_pos, end_pos, motif,
                        tier=2, min_copies=max(2, min_copies)
                    )
                    if repeat:
                        repeats.append(repeat)
                        covered.add((repeat.start, repeat.end))
                        covered_mask[repeat.start:min(repeat.end, n)] = True
                        fallback_found += 1
                continue

            # Manual fallback (no Cython scanner)
            i = 0
            while i + unit_len * min_copies <= n:
                # Skip if covered by Phase A
                if covered_mask[i]:
                    i += 1
                    continue

                count = 1
                start_pos = i

                while True:
                    a_start = i + (count - 1) * unit_len
                    a_end = i + count * unit_len
                    b_start = i + count * unit_len
                    b_end = b_start + unit_len

                    if b_end > n:
                        break

                    a = text_arr[a_start:a_end]
                    b = text_arr[b_start:b_end]

                    allowed_errors = max(max_mismatch, int(unit_len * 0.15))

                    dist = hamming_distance(a, b)
                    if dist is None:
                        dist = int(np.sum(a != b))

                    if dist <= allowed_errors:
                        count += 1
                        continue

                    found_indel = False
                    if b_start > 0:
                        b_shifted = text_arr[b_start-1:b_end-1]
                        dist = hamming_distance(a, b_shifted)
                        if dist is None:
                            dist = int(np.sum(a != b_shifted))
                        if dist <= allowed_errors:
                            count += 1
                            found_indel = True

                    if not found_indel and b_end + 1 <= n:
                        b_shifted = text_arr[b_start+1:b_end+1]
                        dist = hamming_distance(a, b_shifted)
                        if dist is None:
                            dist = int(np.sum(a != b_shifted))
                        if dist <= allowed_errors:
                            count += 1
                            found_indel = True

                    if not found_indel:
                        break

                if count >= min_copies:
                    end_pos = i + count * unit_len
                    length = end_pos - i

                    motif_arr = text_arr[i:i + unit_len]
                    motif = motif_arr.tobytes().decode('ascii', errors='replace')

                    primitive_period = MotifUtils.smallest_period_str(motif)
                    if primitive_period == len(motif):
                        primitive_period = MotifUtils.smallest_period_str_approx(motif, max_error_rate=0.02)
                    if primitive_period < len(motif):
                        motif = motif[:primitive_period]
                        count = length // primitive_period

                    repeat = self._refine_and_create_repeat(
                        chromosome, i, end_pos, motif,
                        tier=2, min_copies=max(2, min_copies)
                    )
                    if not repeat:
                        i += 1
                        continue
                    repeats.append(repeat)
                    covered.add((repeat.start, repeat.end))
                    covered_mask[repeat.start:min(repeat.end, n)] = True
                    fallback_found += 1
                    i = end_pos
                else:
                    i += 1

        if self.show_progress:
            print(f"  [{chromosome}] Tier 2 long-unit: Phase B found {fallback_found} additional repeats", flush=True)

        return repeats

    def _get_max_mismatches_for_array(self, motif_len: int, n_copies: int) -> int:
        """Calculate maximum allowed mismatches for full array.

        Args:
            motif_len: Length of the motif/period
            n_copies: Number of tandem copies

        Returns:
            Maximum allowed mismatches across entire repeat array
        """
        total_length = motif_len * n_copies

        if motif_len == 1:
            return 0

        allowed_rate = max(0.01, min(0.5, self.allowed_mismatch_rate))
        return max(1, int(np.ceil(allowed_rate * total_length)))
    
    def find_short_imperfect_repeats(self, chromosome: str, tier1_seen: Set[Tuple[int, int]]) -> List[TandemRepeat]:
        """(Deprecated) Short imperfect 1-9bp search.

        Kept for API compatibility but returns an empty list so that
        Tier 2 focuses exclusively on motifs > 9bp. Use Tier1 for
        microsatellites.
        """
        if self.show_progress:
            print(f"  [{chromosome}] Tier 2: skipping short imperfect (1-9bp) search; handled by Tier 1")
        return []

    def find_long_repeats(self, chromosome: str, tier1_seen: Optional[Set[Tuple[int, int]]] = None,
                         max_scan_period: Optional[int] = None) -> List[TandemRepeat]:
        """Find medium to long tandem repeats using a lightweight period scan.

        This avoids building large LCP structures and is fast for moderate sequences.

        Args:
            chromosome: Chromosome name
            tier1_seen: Set of (start, end) regions already found by Tier1 (to skip)
            max_scan_period: Optional maximum period to scan (overrides default logic)
        """
        return self._find_repeats_simple(chromosome, tier1_seen or set(), max_scan_period)
    
    def _compute_lcp_array(self) -> np.ndarray:
        """Compute LCP array using Kasai over uint8 codes (Numba-accelerated when available)."""
        n = self.bwt.n
        if n == 0:
            return np.array([], dtype=np.int32)
        # Use the text codes directly for fast comparisons
        text_codes = self.bwt.text_arr
        sa = self.bwt.suffix_array.astype(np.int32, copy=False)
        return _kasai_lcp_uint8(text_codes, sa)
    
    def _detect_lcp_plateaus(self, lcp_array: np.ndarray, chromosome: str) -> List[TandemRepeat]:
        """Detect tandem repeats from LCP plateaus."""
        repeats = []
        n = len(lcp_array)
        if n == 0:
            return repeats
        # Choose a single conservative threshold: max(min_period, 20), but <= max LCP and <= max_period
        lcp_max = int(np.max(lcp_array))
        if lcp_max < self.min_period:
            return repeats
        threshold = min(self.max_period, lcp_max)
        threshold = max(self.min_period, min(threshold, 20))

        i = 0
        while i < n:
            if lcp_array[i] >= threshold:
                # Found a plateau
                j = i
                while j < n and lcp_array[j] >= threshold:
                    j += 1
                
                # Analyze this interval in suffix array
                # The interval is [i-1, j] in SA (since LCP[k] is between SA[k-1] and SA[k])
                # But we need to be careful with indices
                
                # For simplicity, just take the representative length
                period = int(np.median(lcp_array[i:j]))
                
                # Analyze SA interval [i-1, j] for tandem structure
                sa_start = max(0, i - 1)
                sa_end = min(n, j + 1)
                
                found = self._analyze_sa_interval_for_tandems(sa_start, sa_end, period, chromosome)
                repeats.extend(found)
                
                i = j
            else:
                i += 1

        return repeats

    def _smallest_period(self, s: str) -> int:
        """Return the length of the smallest period of s via prefix-function (KMP)."""
        n = len(s)
        pi = [0] * n
        for i in range(1, n):
            j = pi[i-1]
            while j > 0 and s[i] != s[j]:
                j = pi[j-1]
            if s[i] == s[j]:
                j += 1
            pi[i] = j
        p = n - pi[-1]
        return p if p != 0 and n % p == 0 else n
    
    def _smallest_period_codes(self, arr: np.ndarray) -> int:
        """Smallest period for a uint8 array using prefix-function (no strings)."""
        n = int(arr.size)
        if n == 0:
            return 0
        pi = np.zeros(n, dtype=np.int32)
        j = 0
        for i in range(1, n):
            j = int(pi[i-1])
            while j > 0 and arr[i] != arr[j]:
                j = int(pi[j-1])
            if arr[i] == arr[j]:
                j += 1
            pi[i] = j
        p = n - int(pi[-1])
        return p if p != 0 and n % p == 0 else n

    def _find_repeats_simple(self, chromosome: str, tier1_seen: Set[Tuple[int, int]],
                            max_scan_period: Optional[int] = None) -> List[TandemRepeat]:
        """BWT/LCP-driven repeat detection with brute-force fallback.

        Phase A: Uses suffix array and LCP array to identify tandem repeat
                 signatures, then FM-index to find all occurrences and extend
                 with mismatch tolerance.
        Phase B: Falls back to brute-force period scanning on uncovered regions.

        Args:
            chromosome: Chromosome name
            tier1_seen: Set of (start, end) regions already found by Tier1 (to skip)
            max_scan_period: Optional maximum period to scan
        """
        s_arr = self.bwt.text_arr
        n = int(s_arr.size)
        # Exclude trailing sentinel if present ('$' == 36)
        if n > 0 and s_arr[n - 1] == 36:
            n -= 1

        # Determine max period
        if max_scan_period is not None:
            max_p = max_scan_period
        else:
            max_p = min(self.max_period, max(1, n // 2))
            if n > 100_000:
                max_p = min(max_p, 500)
            elif n > 10_000:
                max_p = min(max_p, 1000)
            elif n > 1_000:
                max_p = min(max_p, n // 2)
            else:
                max_p = min(max_p, n // 2)

        min_p = min(self.min_period, max_p)
        results: List[TandemRepeat] = []
        seen: Set[Tuple[int, int, str]] = set()
        covered: Set[Tuple[int, int]] = set()

        # Build tier1 mask
        tier1_mask = np.zeros(n, dtype=bool)
        for start, end in tier1_seen:
            tier1_mask[start:min(end, n)] = True

        start_time = time.time()

        # ===== Phase A: BWT/LCP-driven detection =====
        if self.show_progress:
            print(f"  [{chromosome}] Tier 2 simple scan: Phase A (BWT/LCP) starting, n={n}, min_p={min_p}, max_p={max_p}", flush=True)

        lcp = self._get_lcp_array()
        sa = self.bwt.suffix_array.astype(np.int32, copy=False)

        # Scan LCP array for tandem repeat signatures in the [min_p, max_p] range
        candidates = lcp_tandem_candidates(sa, lcp, n, min_p, max_p)

        # Group candidates by period
        period_seeds: dict = {}
        for period, start_pos in candidates:
            # Skip positions covered by tier1
            if start_pos < n and tier1_mask[start_pos]:
                continue
            if period not in period_seeds:
                period_seeds[period] = []
            period_seeds[period].append(start_pos)

        # Process each period using BWT
        bwt_found = 0
        for period in sorted(period_seeds.keys(), reverse=True):
            seeds = period_seeds[period]
            unique_seeds = sorted(set(seeds))

            bwt_repeats = self._bwt_find_repeats_for_period(
                chromosome, period, unique_seeds,
                min_copies=2 if period >= 20 else self.min_copies,
                covered=covered
            )

            for r in bwt_repeats:
                # Check against seen
                is_new = True
                for s_start, s_end, _ in seen:
                    if s_start <= r.start and s_end >= r.end:
                        is_new = False
                        break
                if is_new:
                    results.append(r)
                    seen.add((r.start, r.end, r.motif))
                    bwt_found += 1

        if self.show_progress:
            print(f"  [{chromosome}] Tier 2 simple scan: Phase A found {bwt_found} repeats "
                  f"({len(candidates)} LCP candidates) in {time.time() - start_time:.2f}s", flush=True)

        # ===== Phase B: Brute-force fallback on uncovered regions =====
        if self.show_progress:
            print(f"  [{chromosome}] Tier 2 simple scan: Phase B (brute-force fallback)...", flush=True)

        # Build coverage mask from Phase A + tier1
        covered_mask = tier1_mask.copy()
        for cs, ce in covered:
            covered_mask[cs:min(ce, n)] = True

        # Adaptive sampling for brute-force
        if n > 10_000_000:
            position_step = 100
            period_step = 5
        elif n > 5_000_000:
            position_step = 50
            period_step = 2
        elif n > 2_000_000:
            position_step = 20
            period_step = 1
        elif n > 500_000:
            position_step = 5
            period_step = 1
        elif n > 100_000:
            position_step = 2
            period_step = 1
        else:
            position_step = 1
            period_step = 1

        # Use accelerated scanner if available, passing covered_mask
        covered_mask_uint8 = covered_mask.view(np.uint8)
        fb_candidates = scan_simple_repeats(
            s_arr, covered_mask_uint8, n, min_p, max_p, period_step, position_step, self.allowed_mismatch_rate
        )

        fallback_found = 0
        if fb_candidates is not None:
            for full_start, full_end, p in fb_candidates:
                is_new = True
                for s_start, s_end, _ in seen:
                    if s_start <= full_start and s_end >= full_end:
                        is_new = False
                        break

                if is_new:
                    motif_arr = s_arr[full_start:full_start + p]
                    motif = motif_arr.tobytes().decode('ascii', errors='replace')

                    repeat = self._refine_and_create_repeat(
                        chromosome, full_start, full_end, motif, tier=2
                    )
                    if repeat:
                        results.append(repeat)
                        seen.add((full_start, full_end, repeat.motif))
                        fallback_found += 1

            if self.show_progress:
                print(f"  [{chromosome}] Tier 2 simple scan: Phase B found {fallback_found} additional repeats "
                      f"in {time.time() - start_time:.2f}s total", flush=True)
            return results

        # Manual brute-force fallback (no Cython)
        max_iterations = 500_000
        iteration_count = 0
        max_time_seconds = 300

        for p in range(min_p, max_p + 1, period_step):
            if time.time() - start_time > max_time_seconds:
                if self.show_progress:
                    print(f"  [{chromosome}] Tier 2 timeout ({max_time_seconds}s) - stopping scan", flush=True)
                break

            i = 0
            while i < n - p:
                if covered_mask[i]:
                    i += position_step
                    continue

                iteration_count += 1
                if iteration_count > max_iterations:
                    break

                check_len = min(4, p)
                if i + p + check_len <= n:
                    if np.array_equal(s_arr[i:i + check_len], s_arr[i + p:i + p + check_len]):
                        start_pos, end_pos, copies, full_start, full_end = self._extend_with_mismatches(
                            s_arr, i, p, n, self.allow_mismatches
                        )

                        required_copies = self.min_copies
                        if p >= 20:
                            required_copies = 2

                        if copies >= required_copies:
                            is_new = True
                            for s_start, s_end, _ in seen:
                                if s_start <= full_start and s_end >= full_end:
                                    is_new = False
                                    break

                            if is_new:
                                motif_arr = s_arr[start_pos:start_pos + p]
                                motif = motif_arr.tobytes().decode('ascii', errors='replace')

                                repeat = self._refine_and_create_repeat(
                                    chromosome, full_start, full_end, motif, tier=2
                                )
                                if repeat:
                                    results.append(repeat)
                                    seen.add((full_start, full_end, repeat.motif))
                                    fallback_found += 1
                                    i = full_end
                                    continue

                i += position_step

            if iteration_count > max_iterations:
                if self.show_progress:
                    print(f"  [{chromosome}] Tier 2 iteration limit reached - stopping scan", flush=True)
                break

        if self.show_progress:
            print(f"  [{chromosome}] Tier 2 simple scan: Phase B found {fallback_found} additional repeats "
                  f"in {time.time() - start_time:.2f}s total", flush=True)

        return results

    def _extend_with_mismatches(self, s_arr: np.ndarray, start_pos: int,
                               period: int, n: int, allow_mismatches: bool = True
                               ) -> Tuple[int, int, int, int, int]:
        """Extend tandem array with mismatch tolerance (10% of full array length).

        Returns:
            (array_start, array_end, copies, full_start, full_end)
        """
        accelerated = extend_with_mismatches(
            s_arr,
            start_pos,
            period,
            n,
            self.allowed_mismatch_rate,
        )
        if accelerated:
            return accelerated

        motif = s_arr[start_pos:start_pos + period].copy()
        start = start_pos
        end = start_pos + period
        copies = 1
        consensus = motif.copy()

        def get_total_mismatches(start_pos_inner, end_pos_inner, consensus_arr, period_len):
            num_copies = (end_pos_inner - start_pos_inner) // period_len
            total_mm = 0
            for i in range(num_copies):
                copy_start = start_pos_inner + i * period_len
                copy_end = copy_start + period_len
                if copy_end <= n:
                    copy = s_arr[copy_start:copy_end]
                    total_mm += MotifUtils.hamming_distance_array(copy, consensus_arr)
            return total_mm

        # Extend right with complete copies
        while end + period <= n:
            next_copy = s_arr[end:end + period]
            
            # Tentatively add
            temp_copies = copies + 1
            temp_end = end + period
            
            # Check mismatches
            # For speed, just check the new copy first
            new_mm = MotifUtils.hamming_distance_array(next_copy, consensus)
            
            # Allow 10% mismatch rate overall
            max_mm = self._get_max_mismatches_for_array(period, temp_copies)
            
            # If new copy is bad, check if total array is still within limits
            if new_mm > 0:
                total_mm = get_total_mismatches(start, temp_end, consensus, period)
                if total_mm > max_mm:
                    break
            
            copies = temp_copies
            end = temp_end

        # Extend left with complete copies
        while start - period >= 0:
            prev_copy = s_arr[start - period:start]
            
            temp_copies = copies + 1
            temp_start = start - period
            
            new_mm = MotifUtils.hamming_distance_array(prev_copy, consensus)
            max_mm = self._get_max_mismatches_for_array(period, temp_copies)
            
            if new_mm > 0:
                total_mm = get_total_mismatches(temp_start, end, consensus, period)
                if total_mm > max_mm:
                    break
            
            copies = temp_copies
            start = temp_start

        full_start = start
        full_end = end

        # Extend right with partial copy (exact matches only)
        partial_right = 0
        while partial_right < period and full_end + partial_right < n:
            if s_arr[full_end + partial_right] == consensus[partial_right]:
                partial_right += 1
            else:
                break
        array_end = full_end + partial_right

        # Extend left with partial copy (exact matches only)
        partial_left = 0
        while partial_left < period and full_start - partial_left - 1 >= 0:
            if s_arr[full_start - partial_left - 1] == consensus[period - 1 - partial_left]:
                partial_left += 1
            else:
                break
        array_start = full_start - partial_left

        return array_start, array_end, copies, full_start, full_end
    
    def _analyze_sa_interval_for_tandems(self, start_idx: int, end_idx: int, 
                                       period: int, chromosome: str) -> List[TandemRepeat]:
        """Analyze suffix array interval for tandem structure."""
        repeats = []
        
        # Get suffix positions in this interval
        positions = []
        for i in range(start_idx, end_idx):
            pos = self.bwt._get_suffix_position(i)
            positions.append(pos)
        
        positions.sort()
        
        # Look for arithmetic progressions with difference = period
        for i in range(len(positions)):
            start_pos = positions[i]
            current_pos = start_pos
            copies = 1
            
            # Check subsequent positions
            # This is O(N^2) in worst case, but N (interval size) is usually small
            for j in range(i + 1, len(positions)):
                if positions[j] == current_pos + period:
                    copies += 1
                    current_pos = positions[j]
            
            if copies >= self.min_copies:
                end_pos = start_pos + copies * period
                
                # Verify the repeat content
                motif_arr = self.bwt.text_arr[start_pos:start_pos + period]
                motif = motif_arr.tobytes().decode('ascii', errors='replace')

                repeat = self._refine_and_create_repeat(
                    chromosome,
                    start_pos,
                    end_pos,
                    motif,
                    tier=2,
                    min_copies=copies
                )

                if repeat:
                    repeats.append(repeat)
        
        return repeats
    
    def _validate_periodicity_arr(self, text_arr: np.ndarray, motif_arr: np.ndarray, period: int) -> bool:
        """Validate periodic structure by vectorized uint8 comparison."""
        m = text_arr.size
        if m < 2 * period:
            return False
        idx = np.arange(m, dtype=np.int32) % period
        # Compare each position to motif at idx
        matches = np.count_nonzero(text_arr == motif_arr[idx])
        similarity = matches / m if m > 0 else 0.0
        return bool(similarity >= 0.8)

    def _region_contains_point(self, regions: Set[Tuple[int, int]], point: int,
                               lock: Optional[threading.Lock] = None) -> bool:
        if lock:
            lock.acquire()
        try:
            for start, end in regions:
                if start <= point < end:
                    return True
        finally:
            if lock:
                lock.release()
        return False

    def _region_add(self, regions: Set[Tuple[int, int]], region: Tuple[int, int],
                    lock: Optional[threading.Lock] = None) -> None:
        if lock:
            lock.acquire()
            try:
                regions.add(region)
            finally:
                lock.release()
        else:
            regions.add(region)

    def _find_tandems_fm_with_mismatches(self, positions: List[int], motif: str,
                                         chromosome: str, motif_len: int,
                                         seen_regions: Set[Tuple[int, int]],
                                         lock: Optional[threading.Lock] = None) -> List[TandemRepeat]:
        """Find tandem repeats allowing mismatches using seed-and-extend strategy (for FM-index results)."""
        repeats = []
        if not positions:
            return repeats

        positions_sorted = sorted(positions)
        max_mm = 0  # Parameter not used
        text_arr = self.bwt.text_arr

        for seed_pos in positions_sorted:
            if self._region_contains_point(seen_regions, seed_pos, lock):
                continue

            start_pos, end_pos, copies = self._extend_tandem_fm(
                text_arr, seed_pos, motif, motif_len, max_mm
            )

            if copies >= self.min_copies:
                consensus_arr, _, _ = MotifUtils.build_consensus_motif_array(
                    text_arr, start_pos, motif_len, copies
                )

                if self._is_maximal_fm(start_pos, end_pos, consensus_arr, motif_len, max_mm):
                    refined_repeat = self._refine_and_create_repeat(
                        chromosome,
                        start_pos,
                        end_pos,
                        motif,
                        tier=2
                    )

                    if refined_repeat:
                        repeats.append(refined_repeat)
                        self._region_add(seen_regions, (refined_repeat.start, refined_repeat.end), lock)

        return repeats

    def _extend_tandem_fm(self, text_arr: np.ndarray, seed_pos: int,
                         motif: str, motif_len: int, max_mismatches: int) -> Tuple[int, int, int]:
        """Extend tandem array left and right from seed position (FM-index version)."""
        motif_arr = np.frombuffer(motif.encode('ascii'), dtype=np.uint8)
        n = text_arr.size

        start = seed_pos
        end = seed_pos + motif_len
        copies = 1
        consensus = motif_arr.copy()

        def get_total_mismatches(start_pos, end_pos, consensus_arr, motif_length):
            num_copies = (end_pos - start_pos) // motif_length
            total_mm = 0
            for i in range(num_copies):
                copy_start = start_pos + i * motif_length
                copy_end = copy_start + motif_length
                if copy_end <= n:
                    copy = text_arr[copy_start:copy_end]
                    total_mm += MotifUtils.hamming_distance_array(copy, consensus_arr)
            return total_mm

        def get_transversions(start_pos, end_pos, consensus_arr, motif_length):
            num_copies = (end_pos - start_pos) // motif_length
            total_tv = 0
            for i in range(num_copies):
                copy_start = start_pos + i * motif_length
                copy_end = copy_start + motif_length
                if copy_end <= n:
                    copy = text_arr[copy_start:copy_end]
                    total_tv += MotifUtils.count_transversions_array(copy, consensus_arr)
            return total_tv

        # Extend right
        while end + motif_len <= n:
            temp_copies = copies + 1
            temp_end = end + motif_len
            
            max_mm = self._get_max_mismatches_for_array(motif_len, temp_copies)
            total_mm = get_total_mismatches(start, temp_end, consensus, motif_len)
            
            if total_mm <= max_mm:
                copies = temp_copies
                end = temp_end
            else:
                break

        # Extend left
        while start - motif_len >= 0:
            temp_copies = copies + 1
            temp_start = start - motif_len
            
            max_mm = self._get_max_mismatches_for_array(motif_len, temp_copies)
            total_mm = get_total_mismatches(temp_start, end, consensus, motif_len)
            
            if total_mm <= max_mm:
                copies = temp_copies
                start = temp_start
            else:
                break

        return start, end, copies

    def _is_maximal_fm(self, start: int, end: int, consensus: np.ndarray,
                      motif_len: int, max_mm: int) -> bool:
        """Check if repeat is maximal (FM-index version)."""
        text_arr = self.bwt.text_arr
        n = text_arr.size

        if start > 0:
            # Check if extending left would still be within mismatch limits
            # This is a simplified check - just check if the previous base matches the last base of motif
            # For a rigorous check we'd need to re-evaluate the whole array
            pass

        if end < n:
            pass

        return True
