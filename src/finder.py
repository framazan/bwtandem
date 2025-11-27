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

    VALID_TIERS = {"tier1", "tier2", "tier3"}

    def __init__(self, sequence: str, chromosome: str = "chr1",
                 min_period: int = 1, max_period: int = 2000,
                 show_progress: bool = False,
                 enabled_tiers: Optional[Set[str]] = None,
                 min_array_bp: Optional[int] = None,
                 max_array_bp: Optional[int] = None):
        self.sequence = sequence
        self.chromosome = chromosome
        self.min_period = min_period
        self.max_period = max_period
        self.show_progress = show_progress
        self.enabled_tiers = self._normalize_tiers(enabled_tiers)

        self.min_array_bp = max(0, min_array_bp) if min_array_bp else None
        self.max_array_bp = max(0, max_array_bp) if max_array_bp else None
        if self.min_array_bp and self.max_array_bp and self.min_array_bp > self.max_array_bp:
            # Swap to keep bounds consistent
            self.min_array_bp, self.max_array_bp = self.max_array_bp, self.min_array_bp
        
        # Initialize BWT Core
        if show_progress:
            print(f"  [{chromosome}] Building BWT and FM-index...", flush=True)
        t0 = time.time()
        # Ensure sentinel
        if not sequence.endswith('$'):
            sequence += '$'
        
        self.bwt = BWTCore(sequence)
        if show_progress:
            print(f"  [{chromosome}] BWT built in {time.time() - t0:.2f}s", flush=True)

        # Initialize Tiers
        self.tier1 = None
        tier1_min = max(1, min_period)
        tier1_max = min(9, max_period)
        if "tier1" in self.enabled_tiers and tier1_min <= tier1_max:
            self.tier1 = Tier1STRFinder(
                self.bwt.text_arr,
                self.bwt,
                max_motif_length=tier1_max,
                min_motif_length=tier1_min,
                allowed_mismatch_rate=0.2,
                allowed_indel_rate=0.1,
                show_progress=show_progress
            )

        self.tier2 = None
        if "tier2" in self.enabled_tiers:
            self.tier2 = Tier2LCPFinder(
                self.bwt,
                min_period=min_period,
                max_period=max_period,
                allowed_mismatch_rate=0.2,
                allowed_indel_rate=0.1,
                show_progress=show_progress
            )

        self.tier3 = Tier3LongReadFinder(self.bwt) if "tier3" in self.enabled_tiers else None

    def find_all(self) -> List[TandemRepeat]:
        """Execute the full 3-tier finding pipeline."""
        all_repeats = []
        tier1_seen: Set[Tuple[int, int]] = set()
        tier2_seen: Set[Tuple[int, int]] = set()

        # --- Tier 1: Short Perfect Repeats ---
        if self.tier1:
            if self.show_progress:
                print(f"  [{self.chromosome}] Running Tier 1 (Short Perfect)...", flush=True)
            t0 = time.time()
            tier1_repeats = self.tier1.find_strs(self.chromosome)

            accepted = 0
            for r in tier1_repeats:
                if self._register_repeat(r, all_repeats, tier1_seen):
                    accepted += 1

            if self.show_progress:
                print(f"  [{self.chromosome}] Tier 1 found {accepted} repeats in {time.time() - t0:.2f}s", flush=True)
        elif self.show_progress:
            print(f"  [{self.chromosome}] Skipping Tier 1 (disabled or out of requested range)", flush=True)

        # --- Tier 2: Imperfect & Medium Repeats (>=10bp by design) ---
        if self.tier2:
            if self.show_progress:
                print(f"  [{self.chromosome}] Running Tier 2 (Imperfect & Medium)...", flush=True)
            t0 = time.time()
            # 2a. Long unit repeats (strict adjacency, >=20bp units)
            long_unit_repeats = []
            long_kept = 0
            min_unit = max(20, self.min_period)
            if self.max_period >= min_unit:
                long_unit_repeats = self.tier2.find_long_unit_repeats_strict(
                    self.chromosome,
                    min_unit_len=min_unit,
                    max_unit_len=self.max_period,
                    min_copies=2
                )

                for r in long_unit_repeats:
                    is_new = True
                    for start, end in tier1_seen:
                        if not (r.end <= start or r.start >= end):
                            is_new = False
                            break
                    if is_new and self._register_repeat(r, all_repeats, tier2_seen):
                        long_kept += 1

            # 2b. General scanning for medium/long repeats up to requested max
            medium_repeats = []
            medium_kept = 0
            # Force Tier2 to ignore classic microsatellites: start from period 10bp
            scan_lower = max(10, self.min_period)
            scan_upper = min(50, self.max_period)
            if scan_upper >= scan_lower:
                combined_seen = tier1_seen.union(tier2_seen)
                medium_repeats = self.tier2.find_long_repeats(
                    self.chromosome,
                    combined_seen,
                    max_scan_period=scan_upper
                )

                for r in medium_repeats:
                    if self._register_repeat(r, all_repeats, tier2_seen):
                        medium_kept += 1
            if self.show_progress:
                total_kept = long_kept + medium_kept
                print(f"  [{self.chromosome}] Tier 2 processed {total_kept} repeats (>=10bp motifs) in {time.time() - t0:.2f}s", flush=True)
        elif self.show_progress:
            print(f"  [{self.chromosome}] Skipping Tier 2 (disabled)", flush=True)

        # --- Tier 3: Long Reads (Optional/Advanced) ---
        # Currently Tier 2 covers most cases, so Tier 3 is a placeholder for future expansion
        # or very specific long-read logic.
        
        # --- Post-processing ---
        # Sort by position
        all_repeats.sort(key=lambda x: x.start)
        
        # Merge adjacent repeats (unify fragmented motifs)
        all_repeats = self._merge_adjacent_repeats(all_repeats)
        
        # Filter overlaps (keep longest/best)
        final_repeats = self._filter_overlaps(all_repeats)
        final_repeats = [r for r in final_repeats if self._repeat_within_bounds(r)]
        
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

    def _merge_adjacent_repeats(self, repeats: List[TandemRepeat]) -> List[TandemRepeat]:
        """Merge adjacent repeats with the same motif."""
        if not repeats:
            return []
            
        # Sort by start position
        repeats.sort(key=lambda x: x.start)
        
        merged = [repeats[0]]
        
        for current in repeats[1:]:
            prev = merged[-1]
            
            # Check if they are adjacent or overlapping
            # Allow a small gap (e.g., up to period length or 10bp)
            gap = max(0, current.start - prev.end)
            
            # Check if motifs are compatible (same canonical motif)
            motif1 = prev.consensus_motif or prev.motif
            motif2 = current.consensus_motif or current.motif
            
            canon1, strand1 = MotifUtils.get_canonical_motif_stranded(motif1)
            canon2, strand2 = MotifUtils.get_canonical_motif_stranded(motif2)
            
            # Allow merge if canonical motifs match and gap is small
            if canon1 == canon2 and gap <= max(10, len(canon1)):
                # Merge them
                new_start = min(prev.start, current.start)
                new_end = max(prev.end, current.end)
                
                # Update prev
                prev.start = new_start
                prev.end = new_end
                prev.length = new_end - new_start
                prev.copies = prev.length / len(canon1)
                
                # Recompute stats for the merged repeat
                self._recompute_stats(prev)
                
            else:
                merged.append(current)
                
        return merged

    def _recompute_stats(self, repeat: TandemRepeat):
        """Recompute statistics for a repeat (e.g. after merging)."""
        text_arr = self.bwt.text_arr
        motif = repeat.consensus_motif or repeat.motif
        motif_len = len(motif)
        
        if motif_len == 0:
            return

        # Re-derive consensus and mismatch rate from the full merged region
        consensus_arr, mm_rate, max_mm = MotifUtils.build_consensus_motif_array(
            text_arr, repeat.start, motif_len, int(repeat.copies)
        )
        
        if consensus_arr.size > 0:
            consensus_str = consensus_arr.tobytes().decode('ascii', errors='replace')
            repeat.consensus_motif = consensus_str
            repeat.motif = consensus_str # Update motif to new consensus
            repeat.mismatch_rate = mm_rate
            repeat.max_mismatches_per_copy = max_mm
            
            # Recalculate TRF stats
            (percent_matches, percent_indels, score, composition,
             entropy, actual_sequence) = MotifUtils.calculate_trf_statistics(
                text_arr, repeat.start, repeat.end, consensus_str, int(repeat.copies), mm_rate
            )
            
            repeat.percent_matches = percent_matches
            repeat.percent_indels = percent_indels
            repeat.score = score
            repeat.composition = composition
            repeat.entropy = entropy
            repeat.actual_sequence = actual_sequence
            
            # Update variations
            variations = MotifUtils.summarize_variations_array(
                text_arr, repeat.start, repeat.end, motif_len, consensus_arr
            )
            repeat.variations = variations

    def cleanup(self):
        """Release resources."""
        self.bwt.clear()

    def _normalize_tiers(self, tiers: Optional[Set[str]]) -> Set[str]:
        if not tiers:
            return set(self.VALID_TIERS)

        normalized: Set[str] = set()
        for tier in tiers:
            if not tier:
                continue
            name = tier.strip().lower()
            if name == "all":
                return set(self.VALID_TIERS)
            if name in self.VALID_TIERS:
                normalized.add(name)

        return normalized if normalized else set(self.VALID_TIERS)

    def _register_repeat(self, repeat: TandemRepeat, store: List[TandemRepeat],
                         seen: Optional[Set[Tuple[int, int]]] = None) -> bool:
        if not self._repeat_within_bounds(repeat):
            return False

        store.append(repeat)
        if seen is not None:
            seen.add((repeat.start, repeat.end))
        return True

    def _repeat_within_bounds(self, repeat: TandemRepeat) -> bool:
        motif = repeat.motif or repeat.consensus_motif
        motif_len = len(motif) if motif else 0
        if motif_len <= 0:
            motif_len = max(1, repeat.length)

        if motif_len < self.min_period or motif_len > self.max_period:
            return False

        length = repeat.length if repeat.length else repeat.end - repeat.start

        if self.min_array_bp and length < self.min_array_bp:
            return False
        if self.max_array_bp and length > self.max_array_bp:
            return False

        return True
