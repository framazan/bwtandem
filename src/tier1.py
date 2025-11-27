import numpy as np
from typing import List, Tuple, Set
from .models import TandemRepeat
from .motif_utils import MotifUtils
from .bwt_core import BWTCore
from . import accelerators

class Tier1STRFinder:
    """Tier 1: Short Perfect Tandem Repeat Finder using optimized sliding window (1-9bp).

    Fast method for detecting perfect tandem repeats with adaptive sampling.
    No BWT/FM-index required - just optimized direct scanning with smart skipping.
    """

    def __init__(self, text_arr: np.ndarray, bwt_core: BWTCore, max_motif_length: int = 9,
                 min_motif_length: int = 1,
                 allowed_mismatch_rate: float = 0.2, allowed_indel_rate: float = 0.1,
                 show_progress: bool = False):
        self.text_arr = text_arr
        self.bwt = bwt_core # Added bwt_core reference as it was used in original code
        self.max_motif_length = max(1, max_motif_length)
        self.min_motif_length = max(1, min(min_motif_length, self.max_motif_length))
        self.min_copies = 3  # Require at least 3 copies to reduce noise
        self.min_array_length = 6  # Minimum total array length in bp
        self.min_entropy = 1.0  # Minimum Shannon entropy to avoid low-complexity
        self.allowed_mismatch_rate = max(0.0, allowed_mismatch_rate)
        self.allowed_indel_rate = max(0.0, allowed_indel_rate)
        self.show_progress = show_progress

    def _build_repeat(self, chromosome: str, refined, tier: int = 1) -> TandemRepeat:
        return MotifUtils.refined_to_repeat(chromosome, refined, tier, self.text_arr, strand='+')

    def _get_max_mismatches_for_array(self, motif_len: int, n_copies: int) -> int:
        """Calculate maximum allowed mismatches for full array.

        Args:
            motif_len: Length of the motif/period
            n_copies: Number of tandem copies

        Returns:
            Maximum allowed mismatches across entire repeat array
        """
        total_length = motif_len * n_copies

        # For single nucleotide repeats (homopolymers), keep strict handling
        if motif_len == 1:
            return 0

        allowed_rate = max(0.01, min(0.5, self.allowed_mismatch_rate))
        return max(1, int(np.ceil(allowed_rate * total_length)))
    
    def _find_simple_tandems_kmer(self, chromosome: str) -> List[TandemRepeat]:
        """Find simple perfect tandem repeats using optimized sliding window.

        This uses a FAST sliding window approach with early skipping.
        No k-mer index needed - direct scanning is faster for tandem repeat detection.
        """
        repeats = []
        text_arr = self.text_arr
        sequence_str = text_arr.tobytes().decode('ascii', errors='replace')
        n = text_arr.size
        seen_regions: Set[Tuple[int, int]] = set()

        # Create bitmap for fast region checking (O(1) instead of O(n))
        seen_mask = np.zeros(n, dtype=bool)

        # For very large chromosomes, use adaptive sampling
        if n > 10_000_000:  # > 10 Mbp
            position_step = 50  # Skip positions for speed
            if self.show_progress:
                print(f"  [{chromosome}] Large sequence ({n:,} bp) - using fast sampling mode (step={position_step})")
        elif n > 5_000_000:  # > 5 Mbp
            position_step = 20
        else:
            position_step = 1

        # Process each motif length (1-9bp) in REVERSE order (longest first)
        # This ensures we detect [AT]n before [A]n, [GCG]n before [GC]n, etc.
        max_len = min(self.max_motif_length, 9)
        min_len = max(1, self.min_motif_length)
        if min_len > max_len:
            return repeats

        for motif_len in range(max_len, min_len - 1, -1):
            i = 0
            while i < n - motif_len:
                # Skip if already in a found region (O(1) bitmap check)
                if seen_mask[i]:
                    i += position_step
                    continue

                # Get potential motif
                motif_arr = text_arr[i:i + motif_len]
                motif = motif_arr.tobytes().decode('ascii', errors='replace')

                # Skip if contains N or other non-ACGT
                if not all(c in 'ACGT' for c in motif):
                    i += position_step
                    continue

                # Count consecutive perfect repeats
                copies = 1
                check_pos = i + motif_len

                while check_pos + motif_len <= n:
                    next_motif_arr = text_arr[check_pos:check_pos + motif_len]
                    if np.array_equal(motif_arr, next_motif_arr):
                        copies += 1
                        check_pos += motif_len
                    else:
                        break

                # If enough copies found, create a repeat
                if copies >= self.min_copies:
                    end_pos = i + copies * motif_len
                    length = end_pos - i

                    # Check entropy
                    entropy = MotifUtils.calculate_entropy(motif)
                    if entropy < self.min_entropy and length < 10:
                        i += position_step
                        continue

                    if length >= self.min_array_length:
                        refined = MotifUtils.refine_repeat(
                            sequence_str,
                            i,
                            end_pos,
                            motif,
                            mismatch_fraction=self.allowed_mismatch_rate,
                            indel_fraction=self.allowed_indel_rate,
                            min_copies=self.min_copies
                        )

                        if not refined:
                            i += position_step
                            continue

                        repeat = self._build_repeat(chromosome, refined, tier=1)
                        repeats.append(repeat)
                        seen_regions.add((refined.start, refined.end))
                        # Update bitmap
                        seen_mask[refined.start:refined.end] = True
                        i = refined.end  # Jump past the repeat
                        continue

                i += position_step

        return repeats

    def find_strs(self, chromosome: str) -> List[TandemRepeat]:
        """Find perfect short tandem repeats (1-9bp) using optimized sliding window.

        This is Tier 1: fast sliding window with adaptive sampling, perfect match only.
        """
        return self._find_simple_tandems_kmer(chromosome)

    def _find_tandems_with_mismatches(self, positions: List[int], motif: str,
                                     chromosome: str, motif_len: int,
                                     seen_regions: Set[Tuple[int, int]]) -> List[TandemRepeat]:
        """Find tandem repeats allowing mismatches using seed-and-extend strategy."""
        repeats = []
        if not positions:
            return repeats

        positions.sort()
        max_mm = 0  # Parameter not used by _extend_tandem_array (kept for backward compatibility)
        text_arr = self.bwt.text_arr
        sequence_str = text_arr.tobytes().decode('ascii', errors='replace')

        for seed_pos in positions:
            # Skip if this position is already part of a found repeat
            if any(start <= seed_pos < end for start, end in seen_regions):
                continue

            # Skip seeds whose window would exceed the chromosome boundary
            if seed_pos + motif_len > text_arr.size:
                continue

            # Explore all rotations that could anchor this seed by sliding up to one motif length
            best_start = None
            best_end = None
            best_copies = 0

            max_left_shift = min(motif_len, seed_pos + 1)
            for shift in range(max_left_shift):
                candidate_seed = seed_pos - shift
                candidate_end = candidate_seed + motif_len

                if candidate_seed < 0 or candidate_end > text_arr.size:
                    continue

                # Skip if candidate seed already claimed by another repeat
                if any(start <= candidate_seed < end for start, end in seen_regions):
                    continue

                candidate_arr = text_arr[candidate_seed:candidate_end]
                candidate_motif = candidate_arr.tobytes().decode('ascii', errors='replace')

                start_pos_candidate, end_pos_candidate, copies_candidate = self._extend_tandem_array(
                    text_arr, candidate_seed, candidate_motif, motif_len, max_mm
                )

                # Ensure the original seed is covered by this candidate extension
                if not (start_pos_candidate <= seed_pos < end_pos_candidate):
                    continue

                if (copies_candidate > best_copies or
                        (copies_candidate == best_copies and
                         (best_start is None or start_pos_candidate < best_start))):
                    best_start = start_pos_candidate
                    best_end = end_pos_candidate
                    best_copies = copies_candidate

            if best_start is None:
                continue

            start_pos, end_pos, copies = best_start, best_end, best_copies

            array_length = end_pos - start_pos

            if copies >= self.min_copies and array_length >= self.min_array_length:
                motif_template = text_arr[start_pos:start_pos + motif_len].tobytes().decode('ascii', errors='replace')
                if not motif_template or not all(c in 'ACGT' for c in motif_template):
                    continue
                refined = MotifUtils.refine_repeat(
                    sequence_str,
                    start_pos,
                    end_pos,
                    motif_template,
                    mismatch_fraction=self.allowed_mismatch_rate,
                    indel_fraction=self.allowed_indel_rate,
                    min_copies=self.min_copies
                )

                if not refined:
                    continue

                repeat = self._build_repeat(chromosome, refined, tier=1)
                repeats.append(repeat)
                seen_regions.add((refined.start, refined.end))

        return repeats

    def _extend_tandem_array(self, text_arr: np.ndarray, seed_pos: int,
                            motif: str, motif_len: int, max_mismatches: int) -> Tuple[int, int, int]:
        """Extend tandem array left and right from seed position allowing mismatches.

        Note: max_mismatches parameter is no longer used; kept for backward compatibility.
        Mismatch tolerance is now calculated as 10% of full array length.

        Returns:
            (start_pos, end_pos, copy_count)
        """
        motif_arr = np.frombuffer(motif.encode('ascii'), dtype=np.uint8)
        n = text_arr.size
        accelerated = accelerators.extend_with_mismatches(
            text_arr,
            seed_pos,
            motif_len,
            n,
            self.allowed_mismatch_rate,
        )
        if accelerated:
            _, _, copies_acc, full_start, full_end = accelerated
            return full_start, full_end, copies_acc

        # Start from seed position
        start = seed_pos
        end = seed_pos + motif_len
        copies = 1

        # Build initial consensus from first copy
        consensus = motif_arr.copy()

        # Helper function to calculate total mismatches across all copies
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

        # Extend right
        while end + motif_len <= n:
            next_copy = text_arr[end:end + motif_len]

            # Tentatively add this copy
            temp_copies = copies + 1
            temp_end = end + motif_len

            # Collect all copies including the new one
            all_copies = []
            for i in range(temp_copies):
                copy_start = start + i * motif_len
                copy_end = copy_start + motif_len
                if copy_end <= n:
                    all_copies.append(text_arr[copy_start:copy_end])

            # Build temporary consensus
            temp_consensus = np.zeros(motif_len, dtype=np.uint8)
            for pos in range(motif_len):
                bases = [copy[pos] for copy in all_copies if pos < len(copy)]
                if bases:
                    unique, counts = np.unique(bases, return_counts=True)
                    temp_consensus[pos] = unique[np.argmax(counts)]

            # Calculate total mismatches with new copy
            total_mm = get_total_mismatches(start, temp_end, temp_consensus, motif_len)
            total_length = temp_copies * motif_len
            max_mm_for_array = self._get_max_mismatches_for_array(motif_len, temp_copies)

            if total_mm <= max_mm_for_array:
                # Accept the new copy
                copies = temp_copies
                end = temp_end
                consensus = temp_consensus
            else:
                break

        # Extend left
        while start - motif_len >= 0:
            prev_copy = text_arr[start - motif_len:start]

            # Tentatively add this copy
            temp_copies = copies + 1
            temp_start = start - motif_len

            # Collect all copies including the new one
            all_copies = []
            for i in range(temp_copies):
                copy_start = temp_start + i * motif_len
                copy_end = copy_start + motif_len
                if copy_end <= n:
                    all_copies.append(text_arr[copy_start:copy_end])

            # Build temporary consensus
            temp_consensus = np.zeros(motif_len, dtype=np.uint8)
            for pos in range(motif_len):
                bases = [copy[pos] for copy in all_copies if pos < len(copy)]
                if bases:
                    unique, counts = np.unique(bases, return_counts=True)
                    temp_consensus[pos] = unique[np.argmax(counts)]

            # Calculate total mismatches with new copy
            total_mm = get_total_mismatches(temp_start, end, temp_consensus, motif_len)
            total_length = temp_copies * motif_len
            max_mm_for_array = self._get_max_mismatches_for_array(motif_len, temp_copies)

            if total_mm <= max_mm_for_array:
                # Accept the new copy
                copies = temp_copies
                start = temp_start
                consensus = temp_consensus
            else:
                break

        return start, end, copies

    def _is_maximal_repeat_approx(self, start: int, end: int, consensus: np.ndarray,
                                  motif_len: int, max_mismatches: int) -> bool:
        """Check if repeat is maximal (cannot be extended) with mismatch tolerance."""
        text_arr = self.bwt.text_arr
        n = text_arr.size

        # Check left extension
        if start >= motif_len:
            left_copy = text_arr[start - motif_len:start]
            if MotifUtils.hamming_distance_array(left_copy, consensus) <= max_mismatches:
                return False

        # Check right extension
        if end + motif_len <= n:
            right_copy = text_arr[end:end + motif_len]
            if MotifUtils.hamming_distance_array(right_copy, consensus) <= max_mismatches:
                return False

        return True
    
    def _find_tandems_in_positions(self, positions: List[int], motif: str, 
                                 chromosome: str, motif_len: int) -> List[TandemRepeat]:
        """Find tandem repeats from motif positions."""
        repeats = []
        if not positions:
            return repeats
        
        positions.sort()
        i = 0
        
        while i < len(positions):
            start_pos = positions[i]
            copies = 1
            current_pos = start_pos
            
            # Extend run of consecutive tandem copies
            j = i + 1
            while j < len(positions):
                expected_pos = current_pos + motif_len
                if positions[j] == expected_pos:
                    copies += 1
                    current_pos = positions[j]
                    j += 1
                else:
                    break
            
            if copies >= self.min_copies:
                # Check maximality
                end_pos = start_pos + copies * motif_len
                if self._is_maximal_repeat(start_pos, end_pos, motif, motif_len):
                    repeat = TandemRepeat(
                        chrom=chromosome,
                        start=start_pos,
                        end=end_pos,
                        motif=motif,
                        copies=copies,
                        length=copies * motif_len,
                        tier=1,
                        confidence=1.0
                    )
                    repeats.append(repeat)
            
            i = j if j > i + 1 else i + 1
        
        return repeats
    
    def _is_maximal_repeat(self, start: int, end: int, motif: str, motif_len: int) -> bool:
        """Check if repeat is maximal (cannot be extended)."""
        # Check left extension
        if start > 0:
            left_char = self.bwt.text[start - 1]
            expected_left = motif[-1]  # Last char of motif
            if left_char == expected_left:
                return False
        
        # Check right extension
        if end < len(self.bwt.text):
            right_char = self.bwt.text[end]
            expected_right = motif[0]  # First char of motif
            if right_char == expected_right:
                return False
        
        return True
