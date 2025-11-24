import time
import numpy as np
from typing import List, Tuple, Set, Optional, Dict
from .bwt_core import BWTCore
from .models import TandemRepeat
from .tier1 import Tier1STRFinder
from .tier2 import Tier2LCPFinder
from .tier3 import Tier3LongReadFinder
from .motif_utils import MotifUtils

class TandemRepeatFinder:
    """Main coordinator for multi-tier tandem repeat finding."""

    def __init__(self, sequence: str, chromosome: str = "chr1",
                 min_period: int = 1, max_period: int = 2000,
                 show_progress: bool = False):
        self.sequence = sequence
        self.chromosome = chromosome
        self.min_period = min_period
        self.max_period = max_period
        self.show_progress = show_progress
        
        # Initialize BWT Core
        if show_progress:
            print(f"  [{chromosome}] Building BWT and FM-index...")
        t0 = time.time()
        # Ensure sentinel
        if not sequence.endswith('$'):
            sequence += '$'
        
        self.bwt = BWTCore(sequence)
        if show_progress:
            print(f"  [{chromosome}] BWT built in {time.time() - t0:.2f}s")

        # Initialize Tiers
        self.tier1 = Tier1STRFinder(self.bwt.text_arr, self.bwt, max_motif_length=9, show_progress=show_progress)
        self.tier2 = Tier2LCPFinder(self.bwt, min_period=min_period, max_period=max_period, show_progress=show_progress)
        self.tier3 = Tier3LongReadFinder(self.bwt)

    def find_all(self) -> List[TandemRepeat]:
        """Execute the full 3-tier finding pipeline."""
        all_repeats = []
        tier1_seen: Set[Tuple[int, int]] = set()
        tier2_seen: Set[Tuple[int, int]] = set()

        # --- Tier 1: Short Perfect Repeats ---
        if self.show_progress:
            print(f"  [{self.chromosome}] Running Tier 1 (Short Perfect)...")
        t0 = time.time()
        tier1_repeats = self.tier1.find_strs(self.chromosome)
        
        for r in tier1_repeats:
            all_repeats.append(r)
            tier1_seen.add((r.start, r.end))
            
        if self.show_progress:
            print(f"  [{self.chromosome}] Tier 1 found {len(tier1_repeats)} repeats in {time.time() - t0:.2f}s")

        # --- Tier 2: Imperfect & Medium Repeats ---
        if self.show_progress:
            print(f"  [{self.chromosome}] Running Tier 2 (Imperfect & Medium)...")
        t0 = time.time()
        
        # 2a. Short imperfect repeats (using FM-index)
        short_imperfect = self.tier2.find_short_imperfect_repeats(self.chromosome, tier1_seen)
        for r in short_imperfect:
            all_repeats.append(r)
            tier2_seen.add((r.start, r.end))
            
        # 2b. Long unit repeats (strict adjacency)
        long_unit_repeats = self.tier2.find_long_unit_repeats_strict(self.chromosome)
        for r in long_unit_repeats:
            # Check overlap with existing
            is_new = True
            for start, end in tier1_seen:
                if not (r.end <= start or r.start >= end):
                    is_new = False
                    break
            if is_new:
                all_repeats.append(r)
                tier2_seen.add((r.start, r.end))

        # 2c. General scanning for medium/long repeats
        # Combine seen sets for masking
        combined_seen = tier1_seen.union(tier2_seen)
        medium_repeats = self.tier2.find_long_repeats(self.chromosome, combined_seen)
        
        for r in medium_repeats:
            all_repeats.append(r)
            tier2_seen.add((r.start, r.end))

        if self.show_progress:
            print(f"  [{self.chromosome}] Tier 2 found {len(short_imperfect) + len(long_unit_repeats) + len(medium_repeats)} repeats in {time.time() - t0:.2f}s")

        # --- Tier 3: Long Reads (Optional/Advanced) ---
        # Currently Tier 2 covers most cases, so Tier 3 is a placeholder for future expansion
        # or very specific long-read logic.
        
        # --- Post-processing ---
        # Sort by position
        all_repeats.sort(key=lambda x: x.start)
        
        # Filter overlaps (keep longest/best)
        final_repeats = self._filter_overlaps(all_repeats)
        
        return final_repeats

    def _filter_overlaps(self, repeats: List[TandemRepeat]) -> List[TandemRepeat]:
        """Filter overlapping repeats, keeping the one with higher score/length."""
        if not repeats:
            return []
            
        # Sort by start position
        repeats.sort(key=lambda x: x.start)
        
        filtered = [repeats[0]]
        
        for current in repeats[1:]:
            prev = filtered[-1]
            
            # Check overlap
            if current.start < prev.end:
                # Calculate overlap amount
                overlap = min(prev.end, current.end) - max(prev.start, current.start)
                overlap_ratio = overlap / min(prev.length, current.length)
                
                if overlap_ratio > 0.5:  # Significant overlap
                    # Keep the better one
                    # Criteria: Length * (1 - mismatch_rate)
                    prev_score = prev.length * (1.0 - prev.mismatch_rate)
                    curr_score = current.length * (1.0 - current.mismatch_rate)
                    
                    if curr_score > prev_score:
                        filtered[-1] = current
                else:
                    # Small overlap, keep both (maybe compound?)
                    filtered.append(current)
            else:
                filtered.append(current)
                
        return filtered

    def cleanup(self):
        """Release resources."""
        self.bwt.clear()
