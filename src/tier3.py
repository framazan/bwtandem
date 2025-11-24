import numpy as np
from typing import List, Tuple, Set, Optional
from .models import TandemRepeat
from .motif_utils import MotifUtils
from .bwt_core import BWTCore

class Tier3LongReadFinder:
    """Tier 3: Long-read repeat finder using seed-and-extend.

    Optimized for finding very long repeats (100bp+) that might have
    significant variation or structural changes.
    """

    def __init__(self, bwt_core: BWTCore, min_length: int = 100,
                 max_length: int = 10000, min_copies: float = 2.0):
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
        3. Check for periodicity between occurrences
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
        # For very long repeats, we can sample sparsely
        sample_step = 50
        kmer_size = 20  # Use long k-mers to ensure uniqueness

        i = 0
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
                positions.sort()
                
                # Check for periodicity in positions
                # We look for at least 2 intervals of similar size
                intervals = np.diff(positions)
                
                if len(intervals) > 0:
                    # Find most common interval (approximate period)
                    # Allow some variance
                    valid_intervals = intervals[intervals >= self.min_length]
                    
                    if len(valid_intervals) > 0:
                        # Simple clustering of intervals
                        # This is a heuristic
                        pass
                        
                        # For now, this is a placeholder for more advanced logic
                        # The current Tier 2 implementation with adaptive scanning
                        # covers most of what Tier 3 would do, but more efficiently.
                        
            i += sample_step

        return repeats
