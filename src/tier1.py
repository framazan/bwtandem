import numpy as np
import time
from typing import List
from itertools import product
from .models import TandemRepeat
from .motif_utils import MotifUtils
from .bwt_core import BWTCore

class Tier1STRFinder:
    """Tier 1: Short Perfect Tandem Repeat Finder (1-9bp) using pure FM-Index.
    
    This implementation adheres strictly to the algorithmic intent:
    It queries the FM-index for exact counts of all canonical motifs in O(k) time,
    then retrieves global positions for promising motifs and collapses adjacent
    hits algebraically.
    """

    def __init__(self, text_arr: np.ndarray, bwt_core: BWTCore, max_motif_length: int = 9,
                 min_motif_length: int = 1,
                 allowed_mismatch_rate: float = 0.2, allowed_indel_rate: float = 0.1,
                 show_progress: bool = False):
        self.text_arr = text_arr
        self.bwt = bwt_core
        self.max_motif_length = max(1, max_motif_length)
        self.min_motif_length = max(1, min(min_motif_length, self.max_motif_length))
        self.min_copies = 3
        # Baseline constraint: TRF uses ~12bp as practical floor for valid microsatellites
        self.min_array_length = 12
        self.min_entropy = 1.0
        self.allowed_mismatch_rate = max(0.0, allowed_mismatch_rate)
        self.allowed_indel_rate = max(0.0, allowed_indel_rate)
        self.show_progress = show_progress

    def _build_repeat(self, chromosome: str, refined, tier: int = 1) -> TandemRepeat:
        return MotifUtils.refined_to_repeat(chromosome, refined, tier, self.text_arr, strand='+')

    def _enumerate_canonical_motifs(self, k: int, alphabet: str = 'ACGT') -> List[str]:
        """Generate all purely primitive motifs of length k."""
        motifs = []
        for chars in product(alphabet, repeat=k):
            motif = ''.join(chars)
            # Skip non-primitive motifs (e.g., ATAT should be found during k=2, not k=4)
            if MotifUtils.smallest_period_str(motif) < k:
                continue
            motifs.append(motif)
        return motifs

    def find_strs(self, chromosome: str) -> List[TandemRepeat]:
        t0 = time.time()
        text_arr = self.text_arr
        n = text_arr.size
        sequence_str = text_arr.tobytes().decode('ascii', errors='replace')
        repeats = []
        
        max_len = min(self.max_motif_length, 9)
        min_len = max(1, self.min_motif_length)
        if min_len > max_len:
            return repeats

        if self.show_progress:
            print(f"  [{chromosome}] Tier 1 pure FM-index scan (min_k={min_len}, max_k={max_len})...", flush=True)

        seen_mask = np.zeros(n, dtype=bool)

        # Iterate longest -> shortest so dominant macro-repeats claim space first
        for motif_len in range(max_len, min_len - 1, -1):
            if self.show_progress:
                print(f"    Evaluating all 4^{motif_len} canonical {motif_len}-mers...", flush=True)
                
            motifs = self._enumerate_canonical_motifs(motif_len)
            
            # Require biology thresholds dynamically (e.g. `A` requires 12 copies, `AT` requires 6)
            required_threshold = max(self.min_array_length, motif_len * self.min_copies)

            for motif in motifs:
                sp, ep = self.bwt.backward_search(motif)
                if sp == -1:
                    continue
                
                occ_count = ep - sp + 1
                
                # Math check: The motif MUST exist at least roughly enough times to form 1 single array
                if occ_count < max(self.min_copies, required_threshold // motif_len):
                    continue

                # Locate all occurrences globally (O(1) given sa_sample_rate=1)
                positions = self.bwt.locate_positions(motif)
                if not positions:
                    continue
                
                # Sort coordinates sequentially
                positions.sort()
                
                # Collapse continuous sequences
                i = 0
                count_pos = len(positions)
                
                while i < count_pos:
                    start_pos = positions[i]
                    copies = 1
                    current_pos = start_pos
                    
                    j = i + 1
                    while j < count_pos and positions[j] == current_pos + motif_len:
                        copies += 1
                        current_pos = positions[j]
                        j += 1
                        
                    length = copies * motif_len
                    
                    # If this run satisfies the threshold, lock it in
                    if length >= required_threshold:
                        end_pos = start_pos + length
                        
                        # Check midpoint/start occlusion
                        midpoint = (start_pos + end_pos) // 2
                        if not seen_mask[start_pos] and not seen_mask[midpoint]:
                            
                            entropy = MotifUtils.calculate_entropy(motif)
                            if entropy >= self.min_entropy or length >= 20: 
                            
                                refined = MotifUtils.refine_repeat(
                                    sequence_str,
                                    start_pos,
                                    end_pos,
                                    motif,
                                    mismatch_fraction=self.allowed_mismatch_rate,
                                    indel_fraction=self.allowed_indel_rate,
                                    min_copies=self.min_copies
                                )
                                
                                if refined:
                                    repeats.append(self._build_repeat(chromosome, refined, tier=1))
                                    seen_mask[refined.start:refined.end] = True

                    # Move `i` cursor to end of chunk (we technically could rewind slightly for overlapping shifts, 
                    # but refine_repeat handles localized shifts around the anchor!)
                    i = j
                    
        if self.show_progress:
            print(f"  [{chromosome}] Tier 1 FM-index found {len(repeats)} repeats in {time.time() - t0:.2f}s", flush=True)

        return repeats
