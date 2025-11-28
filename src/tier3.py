import numpy as np
from typing import List, Tuple, Set, Optional
from .models import TandemRepeat
from .motif_utils import MotifUtils
from .bwt_core import BWTCore
from .accelerators import find_periodic_runs, extend_with_mismatches

class Tier3LongReadFinder:
    """Tier 3: Long-read repeat finder using seed-and-extend.

    Optimized for finding very long repeats (100bp+) that might have
    significant variation or structural changes.
    """

    def __init__(self, bwt_core: BWTCore, min_length: int = 100,
                 max_length: int = 100000, min_copies: float = 2.0):
        self.bwt = bwt_core
        self.min_length = min_length
        self.max_length = max_length
        self.min_copies = min_copies

    def find_long_repeats(self, chromosome: str, tier1_seen: Set[Tuple[int, int]],
                          tier2_seen: Set[Tuple[int, int]]) -> List[TandemRepeat]:
        """Find long repeats not caught by Tier 1 or Tier 2.

        Uses a heuristic approach:
        1. Sample k-mers at regular intervals
        2. Use FM-index to find other occurrences
        3. Check for periodicity between occurrences using Cython accelerator
        """
        repeats = []
        text_arr = self.bwt.text_arr
        n = text_arr.size
        
        # Skip if sequence is too short
        if n < self.min_length * 2:
            return []

        # Create mask of already found regions
        mask = np.zeros(n, dtype=bool)
        for start, end in tier1_seen:
            mask[start:end] = True
        for start, end in tier2_seen:
            mask[start:end] = True

        # Sampling parameters
        sample_step = 100
        kmer_size = 20  # Use long k-mers to ensure uniqueness

        i = 0
        seen_regions = set()

        while i < n - kmer_size:
            if mask[i]:
                i += sample_step
                continue

            # Extract k-mer
            kmer_arr = text_arr[i:i + kmer_size]
            kmer = kmer_arr.tobytes().decode('ascii', errors='replace')

            # Find occurrences
            positions = self.bwt.locate_positions(kmer)
            
            if len(positions) >= self.min_copies:
                # Convert to numpy array for Cython
                pos_arr = np.array(sorted(positions), dtype=np.int64)
                
                # Find periodic patterns
                # We look for periods between min_length and max_length
                patterns = find_periodic_runs(
                    pos_arr, 
                    self.min_length, 
                    self.max_length, 
                    int(self.min_copies),
                    tolerance_ratio=0.03 # 3% tolerance for long repeats
                )
                
                for start_pos, end_pos, period in patterns:
                    # Check if we've seen this region
                    # end_pos from Cython is the start of the last k-mer
                    # So the full span is roughly start_pos to end_pos + kmer_size
                    
                    # Avoid re-processing same region
                    region_key = (start_pos // 100, period // 10) # Approximate key
                    if region_key in seen_regions:
                        continue
                    
                    # Verify and extend
                    # We use the period found to try to extend
                    # Use extend_with_mismatches
                    
                    res = extend_with_mismatches(
                        text_arr, 
                        start_pos, 
                        period, 
                        n, 
                        allowed_mismatch_rate=0.2
                    )
                    
                    if res:
                        arr_start, arr_end, copies, full_start, full_end = res
                        
                        if copies >= self.min_copies and (full_end - full_start) >= self.min_length:
                            # Create repeat
                            # Use arr_start for motif extraction (guaranteed good region from extend)
                            motif_arr = text_arr[arr_start:arr_start + period]
                            motif = motif_arr.tobytes().decode('ascii', errors='replace')
                            
                            # For ultra-long repeats (>100 copies or >10kb), skip expensive DP refinement
                            # But trim boundaries using the seed position as anchor
                            if copies > 100 or (full_end - full_start) > 10000:
                                # The seed position (start_pos from pattern) is guaranteed to be in the repeat
                                # Extract motif from the seed position, not from arr_start
                                motif_arr = text_arr[start_pos:start_pos + period]
                                motif = motif_arr.tobytes().decode('ascii', errors='replace')
                                
                                # Trim boundaries by checking periodicity from this anchor point
                                true_start = start_pos
                                true_end = start_pos + period
                                
                                # Scan backward from start_pos to find where repeats actually start
                                scan_start = max(0, start_pos - period * 50)  # Look back up to 50 periods
                                pos = start_pos - period
                                while pos >= scan_start:
                                    # Check if this position starts a good copy of the motif
                                    window = text_arr[pos:pos + period]
                                    if window.size == period:
                                        matches = np.sum(window == motif_arr)
                                        if matches / period >= 0.75:  # 75% match threshold
                                            true_start = pos
                                            pos -= period
                                        else:
                                            break
                                    else:
                                        break
                                
                                # Scan forward from start_pos + period to find where repeats actually end
                                scan_end = min(n, start_pos + period * 600)  # Look forward up to 600 periods
                                pos = start_pos + period
                                while pos + period <= scan_end:
                                    window = text_arr[pos:pos + period]
                                    if window.size == period:
                                        matches = np.sum(window == motif_arr)
                                        if matches / period >= 0.75:  # 75% match threshold
                                            true_end = pos + period
                                            pos += period
                                        else:
                                            break
                                    else:
                                        break
                                
                                # Recalculate copies based on true boundaries
                                true_copies = (true_end - true_start) / period
                                
                                if true_copies >= self.min_copies:
                                    # Build consensus from first few copies for efficiency
                                    max_consensus_copies = min(int(true_copies), 20)
                                    consensus_arr, mm_rate, max_mm = MotifUtils.build_consensus_motif_array(
                                        text_arr, true_start, period, max_consensus_copies
                                    )
                                    consensus_motif = consensus_arr.tobytes().decode('ascii', errors='replace') if consensus_arr.size > 0 else motif
                                    
                                    # Calculate TRF stats directly
                                    (percent_matches, percent_indels, score, composition,
                                     entropy, actual_sequence) = MotifUtils.calculate_trf_statistics(
                                        text_arr, true_start, true_end, consensus_motif, int(true_copies), mm_rate
                                    )
                                    
                                    repeat = TandemRepeat(
                                        chrom=chromosome,
                                        start=true_start,
                                        end=true_end,
                                        motif=motif,
                                        copies=float(true_copies),
                                        length=true_end - true_start,
                                        tier=3,
                                        confidence=max(0.5, 1.0 - mm_rate),
                                        consensus_motif=consensus_motif,
                                        mismatch_rate=mm_rate,
                                        max_mismatches_per_copy=max_mm,
                                        n_copies_evaluated=max_consensus_copies,
                                        strand='+',
                                        percent_matches=percent_matches,
                                        percent_indels=percent_indels,
                                        score=score,
                                        composition=composition,
                                        entropy=entropy,
                                        actual_sequence=actual_sequence[:500] if len(actual_sequence) > 500 else actual_sequence,
                                        variations=None
                                    )
                                else:
                                    repeat = None
                            else:
                                # Use full DP refinement for shorter repeats
                                # Refine
                                refined = MotifUtils.refine_repeat(
                                    self.bwt.text,
                                    arr_start,
                                    arr_end,
                                    motif,
                                    mismatch_fraction=0.2,
                                    indel_fraction=0.1,
                                    min_copies=int(self.min_copies)
                                )
                                
                                if refined:
                                    repeat = MotifUtils.refined_to_repeat(chromosome, refined, tier=3, text_arr=text_arr)
                                else:
                                    repeat = None
                            
                            if repeat:
                                    # Check overlap with existing results
                                    is_new = True
                                    for r in repeats:
                                        if r.start <= repeat.start and r.end >= repeat.end:
                                            is_new = False
                                            break
                                    
                                    if is_new:
                                        repeats.append(repeat)
                                        seen_regions.add(region_key)
                                        # Actively mask found region to skip future seeds
                                        mask[repeat.start:repeat.end] = True

            i += sample_step

        return repeats
