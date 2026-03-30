import numpy as np
from typing import List, Tuple, Set, Optional
from .models import TandemRepeat
from .motif_utils import MotifUtils
from .bwt_core import BWTCore
from .accelerators import extend_with_mismatches
from .bwt_seed import bwt_kmer_seed_scan

class Tier3LongReadFinder:
    """Tier 3: Long-read repeat finder using BWT k-mer seeding.

    Uses the shared BWT seeding core (bwt_seed.py) with Tier 3-specific
    parameters (long periods, sparse sampling, high divergence tolerance)
    and Tier 3-specific post-processing (anchor-based boundary verification,
    consensus from sampled copies).
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

        Uses the shared BWT k-mer seeding pipeline with Tier 3 parameters:
        - Large k-mers (20bp) for uniqueness
        - Sparse sampling (stride=100) for speed
        - Wide period range (100bp-100kbp)
        - Higher divergence tolerance (20%)

        Tier 3 post-processing:
        - Anchor-based boundary verification for ultra-long arrays
        - Consensus from sampled copies (not full DP) for efficiency
        """
        text_arr = self.bwt.text_arr
        n = text_arr.size

        # Skip if sequence is too short
        if n < self.min_length * 2:
            return []

        # Create mask of already found regions
        mask = np.zeros(n, dtype=bool)
        for start, end in tier1_seen:
            mask[start:min(end, n)] = True
        for start, end in tier2_seen:
            mask[start:min(end, n)] = True

        # ===== Phase A: Shared BWT k-mer seeding =====
        seed_candidates = bwt_kmer_seed_scan(
            bwt=self.bwt,
            min_period=self.min_length,
            max_period=self.max_length,
            kmer_size=20,          # Long k-mers for uniqueness
            stride=100,            # Sparse sampling for speed
            min_copies=int(self.min_copies),
            allowed_mismatch_rate=0.20,
            tolerance_ratio=0.03,  # 3% tolerance for long repeats
            max_occurrences=500,   # Aggressive cap for Tier 3
            covered_mask=mask,
            show_progress=False,
            label=f"{chromosome} Tier3",
        )

        # ===== Tier 3 post-processing =====
        repeats = []
        seen_regions = set()

        for cand in seed_candidates:
            region_key = (cand.start // max(cand.period, 1), cand.period)
            if region_key in seen_regions:
                continue

            period = cand.period
            copies = cand.copies
            full_start = cand.start
            full_end = cand.end

            # For ultra-long repeats (>100 copies or >10kb), use anchor-based
            # boundary verification instead of expensive DP refinement
            if copies > 100 or (full_end - full_start) > 10000:
                motif_arr = text_arr[cand.seed_pos:cand.seed_pos + period]
                if motif_arr.size < period:
                    motif_arr = text_arr[full_start:full_start + period]
                motif = motif_arr.tobytes().decode('ascii', errors='replace')

                # Anchor-based boundary verification:
                # Scan backward from seed to find true start
                true_start = cand.seed_pos
                true_end = cand.seed_pos + period

                scan_start = max(0, cand.seed_pos - period * 50)
                pos = cand.seed_pos - period
                while pos >= scan_start:
                    window = text_arr[pos:pos + period]
                    if window.size == period:
                        matches = np.sum(window == motif_arr)
                        if matches / period >= 0.75:
                            true_start = pos
                            pos -= period
                        else:
                            break
                    else:
                        break

                # Scan forward from seed to find true end
                scan_end = min(n, cand.seed_pos + period * 600)
                pos = cand.seed_pos + period
                while pos + period <= scan_end:
                    window = text_arr[pos:pos + period]
                    if window.size == period:
                        matches = np.sum(window == motif_arr)
                        if matches / period >= 0.75:
                            true_end = pos + period
                            pos += period
                        else:
                            break
                    else:
                        break

                true_copies = (true_end - true_start) / period

                if true_copies >= self.min_copies:
                    max_consensus_copies = min(int(true_copies), 20)
                    consensus_arr, mm_rate, max_mm = MotifUtils.build_consensus_motif_array(
                        text_arr, true_start, period, max_consensus_copies
                    )
                    consensus_motif = consensus_arr.tobytes().decode('ascii', errors='replace') if consensus_arr.size > 0 else motif

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
                motif_arr = text_arr[full_start:full_start + period]
                motif = motif_arr.tobytes().decode('ascii', errors='replace')

                refined = MotifUtils.refine_repeat(
                    self.bwt.text,
                    full_start,
                    full_end,
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
                    mask[repeat.start:min(repeat.end, n)] = True

        return repeats
