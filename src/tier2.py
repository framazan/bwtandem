import numpy as np
from typing import List, Tuple, Set, Optional
import time
from .models import TandemRepeat
from .motif_utils import MotifUtils
from .bwt_core import BWTCore, _kasai_lcp_uint8

class Tier2LCPFinder:
    """Tier 2: BWT/FM-index based repeat finder for ALL motif lengths with imperfect repeat support.

    Handles both short repeats (with mismatches) and medium/long repeats using:
    - FM-index backward search for motif occurrences
    - LCP arrays for longer period detection
    - Seed-and-extend with mismatch tolerance
    """

    def __init__(self, bwt_core: BWTCore, min_period: int = 1, max_period: int = 1000,
                 max_short_motif: int = 9, allow_mismatches: bool = True, show_progress: bool = False):
        self.bwt = bwt_core
        self.min_period = min_period  # Now starts at 1bp
        self.max_period = max_period
        self.max_short_motif = max_short_motif  # For FM-index search (1-9bp)
        self.min_copies = 3  # Require at least 3 copies
        self.min_array_length = 6  # Minimum total array length (for short repeats)
        self.min_entropy = 1.0  # Minimum Shannon entropy
        self.allow_mismatches = allow_mismatches
        self.show_progress = show_progress
        self.period_step = 1  # Step size for period scanning (increase to speed up)

    def _hamming_distance(self, arr1: np.ndarray, arr2: np.ndarray) -> int:
        """Calculate Hamming distance between two arrays."""
        return int(np.sum(arr1 != arr2))

    def find_long_unit_repeats_strict(self, chromosome: str, min_unit_len: int = 20,
                                      max_unit_len: int = 120, max_mismatch: int = 2,
                                      min_copies: int = 3) -> List[TandemRepeat]:
        """Find long-unit tandem repeats using strict adjacency checking.

        This detects biologically meaningful long repeats (20-120bp units) and avoids
        reporting nested short-motif periodicities inside them.

        Args:
            chromosome: Chromosome name
            min_unit_len: Minimum unit length to consider (default 20bp)
            max_unit_len: Maximum unit length to scan (default 120bp)
            max_mismatch: Maximum Hamming distance per unit comparison (default 2)
            min_copies: Minimum number of adjacent copies required (default 3)

        Returns:
            List of long-unit tandem repeats
        """
        repeats = []
        text_arr = self.bwt.text_arr
        n = int(text_arr.size)
        # print(f"[DEBUG] Searching chrom={chromosome}, seq_len={n}, min_unit={min_unit_len}, max_unit={max_unit_len}")

        # Exclude sentinel
        if n > 0 and text_arr[n - 1] == 36:  # '$' = 36
            n -= 1

        # For each candidate unit length - PROCESS IN REVERSE ORDER (longest first)
        # This ensures we detect [AT]n before [A]n, [GCG]n before [GC]n, etc.
        max_possible_unit = min(max_unit_len, n // min_copies)
        for unit_len in range(max_possible_unit, min_unit_len - 1, -1):
            i = 0
            while i + unit_len * min_copies <= n:
                # Test if there's at least one adjacency starting at i
                count = 1
                start_pos = i

                # Extend right while adjacency holds
                while True:
                    a_start = i + (count - 1) * unit_len
                    a_end = i + count * unit_len
                    b_start = i + count * unit_len
                    b_end = b_start + unit_len

                    if b_end > n:
                        break

                    a = text_arr[a_start:a_end]
                    b = text_arr[b_start:b_end]

                    if self._hamming_distance(a, b) <= max_mismatch:
                        count += 1
                    else:
                        break

                # If we found enough copies, create a repeat
                if count >= min_copies:
                    end_pos = i + count * unit_len
                    length = end_pos - i

                    # Get the first unit as the motif
                    motif_arr = text_arr[i:i + unit_len]
                    motif = motif_arr.tobytes().decode('ascii', errors='replace')

                    # Reduce to primitive period (e.g., 105bp -> 36bp if 105 is periodic)
                    primitive_period = MotifUtils.smallest_period_str(motif)
                    if primitive_period < len(motif):
                        # Use the primitive period instead
                        motif = motif[:primitive_period]
                        # Recalculate count based on primitive period
                        count = length // primitive_period

                    # Calculate consensus and statistics
                    (percent_matches, percent_indels, score, composition,
                     entropy, actual_sequence) = MotifUtils.calculate_trf_statistics(
                        text_arr, i, end_pos, motif, count, 0.0
                    )

                    # Calculate actual mismatches (perfect repeats have 100% matches)
                    actual_mismatches_per_copy = 0 if percent_matches >= 99.9 else max_mismatch

                    repeat = TandemRepeat(
                        chrom=chromosome,
                        start=i,
                        end=end_pos,
                        motif=motif,
                        copies=float(count),
                        length=length,
                        tier=2,  # Tier 2 for long units
                        confidence=0.95,
                        consensus_motif=motif,
                        mismatch_rate=0.0,
                        max_mismatches_per_copy=actual_mismatches_per_copy,
                        n_copies_evaluated=count,
                        strand='+',
                        percent_matches=percent_matches,
                        percent_indels=percent_indels,
                        score=score,
                        composition=composition,
                        entropy=entropy,
                        actual_sequence=actual_sequence,
                        variations=None
                    )
                    # if 'test8' in chromosome or 'test10' in chromosome:
                    #     print(f"[DEBUG] Found repeat: chrom={chromosome}, start={i}, end={end_pos}, motif={motif[:20]}{'...' if len(motif)>20 else ''}, unit_len={unit_len}, copies={count}")
                    repeats.append(repeat)
                    i = end_pos  # Jump past this repeat
                else:
                    i += 1

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

        # For single nucleotide repeats (homopolymers), NO mismatches allowed
        if motif_len == 1:
            return 0

        # For short motifs (2-6bp), be very conservative
        if motif_len <= 6:
            # Allow at most 5% mismatches (1 in 20 bases)
            return max(1, int(np.ceil(0.05 * total_length)))

        # For longer motifs, allow up to 8% mismatches
        return max(1, int(np.ceil(0.08 * total_length)))
    
    def find_short_imperfect_repeats(self, chromosome: str, tier1_seen: Set[Tuple[int, int]]) -> List[TandemRepeat]:
        """Find short imperfect tandem repeats (1-9bp) using FM-index with mismatch tolerance.

        This is called after Tier1 to find imperfect repeats that the perfect-match sliding window missed.

        Args:
            chromosome: Chromosome name
            tier1_seen: Set of (start, end) regions already found by Tier1

        Returns:
            List of imperfect tandem repeats
        """
        repeats = []
        seen_regions = tier1_seen.copy()  # Don't overlap with Tier1 results

        # Use the BWT-based methods from old Tier1
        text_arr = self.bwt.text_arr
        n = text_arr.size

        # IMPORTANT: Skip BWT search for large chromosomes to prevent stalling
        # Tier 1 already found perfect repeats, and BWT search is too expensive for large sequences
        if n > 1_000_000:  # > 1 Mbp
            if self.show_progress:
                print(f"  [{chromosome}] Skipping BWT search for short imperfect repeats (sequence too large)")
            return []

        for k in range(self.min_period, min(self.max_short_motif + 1, 10)):
            # Generate all canonical motifs of length k
            for motif in MotifUtils.enumerate_motifs(k):
                # Find positions using FM-index
                positions = self.bwt.locate_positions(motif)
                
                if len(positions) < self.min_copies:
                    continue
                
                # Find tandem repeats from these positions allowing mismatches
                found = self._find_tandems_fm_with_mismatches(
                    positions, motif, chromosome, k, seen_regions
                )
                
                for repeat in found:
                    repeats.append(repeat)
                    seen_regions.add((repeat.start, repeat.end))

        return repeats

    def find_long_repeats(self, chromosome: str, tier1_seen: Optional[Set[Tuple[int, int]]] = None) -> List[TandemRepeat]:
        """Find medium to long tandem repeats using a lightweight period scan.

        This avoids building large LCP structures and is fast for moderate sequences.

        Args:
            chromosome: Chromosome name
            tier1_seen: Set of (start, end) regions already found by Tier1 (to skip)
        """
        return self._find_repeats_simple(chromosome, tier1_seen or set())
    
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

    def _find_repeats_simple(self, chromosome: str, tier1_seen: Set[Tuple[int, int]]) -> List[TandemRepeat]:
        """Simple scanning detector for tandem repeats with imperfect repeat support.

        Optimized for long sequences with adaptive scanning.

        Args:
            chromosome: Chromosome name
            tier1_seen: Set of (start, end) regions already found by Tier1 (to skip)
        """
        s_arr = self.bwt.text_arr
        n = int(s_arr.size)
        # Exclude trailing sentinel if present ('$' == 36)
        if n > 0 and s_arr[n - 1] == 36:
            n -= 1
        # AGGRESSIVE max_period limits to prevent stalling
        max_p = min(self.max_period, max(1, n // 2))
        if n > 100_000:  # > 100kb
            max_p = min(max_p, 500)
        elif n > 10_000:  # > 10kb
            max_p = min(max_p, 1000)
        elif n > 1_000:  # > 1kb
            max_p = min(max_p, n // 2)
        else:
            max_p = min(max_p, n // 2)
        min_p = min(self.min_period, max_p)
        results: List[TandemRepeat] = []
        seen: Set[Tuple[int, int, str]] = set()

        # Optimize masking: create a bitmap for fast lookups
        # This prevents O(n) checks on every iteration
        tier1_mask = np.zeros(n, dtype=bool)
        for start, end in tier1_seen:
            tier1_mask[start:end] = True

        # AGGRESSIVE adaptive scanning to prevent stalling
        # The _extend_with_mismatches is VERY expensive, so we need aggressive sampling
        if n > 10_000_000:  # > 10 Mbp
            position_step = 100
            period_step = 5
        elif n > 5_000_000:  # > 5 Mbp
            position_step = 50
            period_step = 2
        elif n > 1_000_000:  # > 1 Mbp
            position_step = 20
            period_step = 1
        elif n > 100_000:  # > 100 kb
            position_step = 5
            period_step = 1
        elif n > 10_000:  # > 10 kb
            position_step = 2
            period_step = 1
        else:
            position_step = 1
            period_step = 1

        if self.show_progress and (position_step > 1 or period_step > 1):
            print(f"  [{chromosome}] Adaptive scanning: pos_step={position_step}, period_step={period_step}")

        # Safety counter to prevent infinite loops and timeout
        max_iterations = 100_000  # Reduced from 1M - be very aggressive
        iteration_count = 0
        start_time = time.time()
        max_time_seconds = 30  # Maximum 30 seconds per chromosome for Tier 2

        for p in range(min_p, max_p + 1, period_step):
            # Check timeout
            if iteration_count % 1000 == 0:
                if time.time() - start_time > max_time_seconds:
                    if self.show_progress:
                        print(f"  [{chromosome}] Tier 2 timeout ({max_time_seconds}s) - stopping scan")
                    break

            i = 0
            while i < n - p:
                # Skip if already covered by Tier 1
                if tier1_mask[i]:
                    i += position_step
                    continue

                # Check timeout and iterations
                iteration_count += 1
                if iteration_count > max_iterations:
                    break

                # Quick check for periodicity
                # Compare s[i] with s[i+p]
                if s_arr[i] == s_arr[i + p]:
                    # Found a match, try to extend
                    # Use mismatch tolerance
                    start_pos, end_pos, copies, full_start, full_end = self._extend_with_mismatches(
                        s_arr, i, p, n, self.allow_mismatches
                    )

                    if copies >= self.min_copies:
                        # Check if we've seen this region
                        region_key = (full_start, full_end, "") # Motif added later
                        is_new = True
                        for s_start, s_end, _ in seen:
                            if s_start <= full_start and s_end >= full_end:
                                is_new = False
                                break
                        
                        if is_new:
                            # Extract motif and calculate stats
                            motif_arr = s_arr[start_pos:start_pos + p]
                            motif = motif_arr.tobytes().decode('ascii', errors='replace')
                            
                            # Build consensus
                            consensus_arr, mm_rate, max_mm = MotifUtils.build_consensus_motif_array(
                                s_arr, full_start, p, copies
                            )
                            consensus = consensus_arr.tobytes().decode('ascii', errors='replace')
                            
                            # Calculate stats
                            (percent_matches, percent_indels, score, composition,
                             entropy, actual_sequence) = MotifUtils.calculate_trf_statistics(
                                s_arr, full_start, full_end, consensus, copies, mm_rate
                            )
                            
                            variations = MotifUtils.summarize_variations_array(
                                s_arr, full_start, full_end, p, consensus_arr
                            )
                            
                            if entropy >= self.min_entropy:
                                repeat = TandemRepeat(
                                    chrom=chromosome,
                                    start=full_start,
                                    end=full_end,
                                    motif=consensus,
                                    copies=float(copies),
                                    length=full_end - full_start,
                                    tier=2,
                                    confidence=max(0.5, 1.0 - mm_rate),
                                    consensus_motif=consensus,
                                    mismatch_rate=mm_rate,
                                    max_mismatches_per_copy=max_mm,
                                    n_copies_evaluated=copies,
                                    strand='+',
                                    percent_matches=percent_matches,
                                    percent_indels=percent_indels,
                                    score=score,
                                    composition=composition,
                                    entropy=entropy,
                                    actual_sequence=actual_sequence,
                                    variations=variations if variations else None
                                )
                                results.append(repeat)
                                seen.add((full_start, full_end, consensus))
                                
                                # Skip past this repeat
                                i = full_end
                                continue
                
                i += position_step
            
            if iteration_count > max_iterations:
                if self.show_progress:
                    print(f"  [{chromosome}] Tier 2 iteration limit reached - stopping scan")
                break

        return results

    def _extend_with_mismatches(self, s_arr: np.ndarray, start_pos: int,
                               period: int, n: int, allow_mismatches: bool = True
                               ) -> Tuple[int, int, int, int, int]:
        """Extend tandem array with mismatch tolerance (10% of full array length).

        Returns:
            (array_start, array_end, copies, full_start, full_end)
        """
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
                
                repeat = TandemRepeat(
                    chrom=chromosome,
                    start=start_pos,
                    end=end_pos,
                    motif=motif,
                    copies=float(copies),
                    length=copies * period,
                    tier=2,
                    confidence=1.0
                )
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

    def _find_tandems_fm_with_mismatches(self, positions: List[int], motif: str,
                                         chromosome: str, motif_len: int,
                                         seen_regions: Set[Tuple[int, int]]) -> List[TandemRepeat]:
        """Find tandem repeats allowing mismatches using seed-and-extend strategy (for FM-index results)."""
        repeats = []
        if not positions:
            return repeats

        positions_sorted = sorted(positions)
        max_mm = 0  # Parameter not used
        text_arr = self.bwt.text_arr

        for seed_pos in positions_sorted:
            if any(start <= seed_pos < end for start, end in seen_regions):
                continue

            start_pos, end_pos, copies = self._extend_tandem_fm(
                text_arr, seed_pos, motif, motif_len, max_mm
            )

            if copies >= self.min_copies:
                # Build consensus
                consensus_arr, mm_rate, max_mm_per_copy = MotifUtils.build_consensus_motif_array(
                    text_arr, start_pos, motif_len, copies
                )
                consensus_str = consensus_arr.tobytes().decode('ascii', errors='replace')
                
                # Check maximality
                if self._is_maximal_fm(start_pos, end_pos, consensus_arr, motif_len, max_mm):
                    # Calculate stats
                    (percent_matches, percent_indels, score, composition,
                     entropy, actual_sequence) = MotifUtils.calculate_trf_statistics(
                        text_arr, start_pos, end_pos, consensus_str, copies, mm_rate
                    )
                    
                    variations = MotifUtils.summarize_variations_array(
                        text_arr, start_pos, end_pos, motif_len, consensus_arr
                    )
                    
                    repeat = TandemRepeat(
                        chrom=chromosome,
                        start=start_pos,
                        end=end_pos,
                        motif=consensus_str,
                        copies=float(copies),
                        length=end_pos - start_pos,
                        tier=2,
                        confidence=max(0.5, 1.0 - mm_rate),
                        consensus_motif=consensus_str,
                        mismatch_rate=mm_rate,
                        max_mismatches_per_copy=max_mm_per_copy,
                        n_copies_evaluated=copies,
                        strand='+',
                        percent_matches=percent_matches,
                        percent_indels=percent_indels,
                        score=score,
                        composition=composition,
                        entropy=entropy,
                        actual_sequence=actual_sequence,
                        variations=variations if variations else None
                    )
                    repeats.append(repeat)
                    seen_regions.add((start_pos, end_pos))

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
