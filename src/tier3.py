import math
import numpy as np
from typing import List, Tuple, Set, Optional
from .models import TandemRepeat
from .motif_utils import MotifUtils
from .bwt_core import BWTCore
from .accelerators import anchor_scan_boundaries
from .bwt_seed import bwt_kmer_seed_scan


def compute_adaptive_params(
    seq_len: int,
    gc_content: float,
    coverage_ratio: float,
    min_period: int,
    max_period: int,
    preset: str = "balanced",
) -> dict:
    """Compute adaptive Tier 3 parameters based on input characteristics."""
    # Preset → speed_factor
    speed_weights = {"fast": 0.8, "balanced": 0.5, "sensitive": 0.2}
    speed_weight = speed_weights.get(preset, 0.5)
    speed_factor = speed_weight / 0.5

    # Base formulas (continuous functions)
    safe_seq = max(seq_len, 1)

    base_kmer = int(10 + 6 * math.log10(max(safe_seq / 1e5, 1)))
    base_stride = int(safe_seq / 40000)
    base_max_occ = int(safe_seq / 30000)
    base_scan_bw = int(50 * safe_seq / 1e8)
    base_scan_fw = int(600 * safe_seq / 1e8)

    # Accuracy params (NOT affected by preset)
    allowed_mismatch_rate = 0.15 + 0.10 * abs(gc_content - 0.5)
    tolerance_ratio = 0.02 + 0.02 * (max_period / 100000)
    anchor_match_pct = 0.70 + 0.10 * (1 - coverage_ratio)

    # Apply preset to speed params
    kmer_size = int(base_kmer + (speed_factor - 1) * 2)
    stride = int(base_stride * speed_factor)
    max_occurrences = int(base_max_occ / speed_factor)
    scan_backward = int(base_scan_bw / speed_factor)
    scan_forward = int(base_scan_fw / speed_factor)

    # Coverage correction
    if coverage_ratio > 0.5:
        stride = int(stride * (1 - 0.5 * coverage_ratio))

    # Threshold-based strategy changes
    if safe_seq > 100_000_000:  # large-chr mode
        stride = max(stride, 150)
        kmer_size = max(kmer_size, 20)
        max_occurrences = min(max_occurrences, 500)
    elif safe_seq < 100_000:  # micro mode
        stride = max(stride, 20)
        kmer_size = max(kmer_size, 12)

    # Clamp all values
    kmer_size = max(12, min(28, kmer_size))
    stride = max(20, min(300, stride))
    allowed_mismatch_rate = max(0.15, min(0.20, allowed_mismatch_rate))
    tolerance_ratio = max(0.02, min(0.04, tolerance_ratio))
    max_occurrences = max(200, min(1500, max_occurrences))
    anchor_match_pct = max(0.70, min(0.80, anchor_match_pct))
    scan_backward = max(20, min(80, scan_backward))
    scan_forward = max(200, min(800, scan_forward))

    return {
        "kmer_size": kmer_size,
        "stride": stride,
        "allowed_mismatch_rate": allowed_mismatch_rate,
        "tolerance_ratio": tolerance_ratio,
        "max_occurrences": max_occurrences,
        "anchor_match_pct": anchor_match_pct,
        "scan_backward": scan_backward,
        "scan_forward": scan_forward,
    }


class Tier3LongReadFinder:
    """Tier 3: Long-read repeat finder using BWT k-mer seeding.

    Uses the shared BWT seeding core (bwt_seed.py) with Tier 3-specific
    parameters (long periods, sparse sampling, high divergence tolerance)
    and Tier 3-specific post-processing (anchor-based boundary verification,
    consensus from sampled copies).
    """

    def __init__(self, bwt_core: BWTCore, min_length: int = 100,
                 max_length: int = 100000, min_copies: float = 2.0,
                 mode: str = "balanced"):
        self.bwt = bwt_core
        self.min_length = min_length
        self.max_length = max_length
        self.min_copies = min_copies
        self.mode = mode

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

        # Compute adaptive params based on sequence characteristics
        gc_content = float(np.mean((text_arr == ord('G')) | (text_arr == ord('C'))))
        coverage_ratio = float(np.mean(mask))
        params = compute_adaptive_params(
            seq_len=n,
            gc_content=gc_content,
            coverage_ratio=coverage_ratio,
            min_period=self.min_length,
            max_period=self.max_length,
            preset=self.mode,
        )

        anchor_match_pct = params["anchor_match_pct"]
        scan_bw_periods = params["scan_backward"]
        scan_fw_periods = params["scan_forward"]

        # ===== Phase A: Shared BWT k-mer seeding =====
        seed_candidates = bwt_kmer_seed_scan(
            bwt=self.bwt,
            min_period=self.min_length,
            max_period=self.max_length,
            kmer_size=params["kmer_size"],
            stride=params["stride"],
            min_copies=int(self.min_copies),
            allowed_mismatch_rate=params["allowed_mismatch_rate"],
            tolerance_ratio=params["tolerance_ratio"],
            max_occurrences=params["max_occurrences"],
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
                seed_pos = cand.seed_pos
                # Use seed_pos as motif source; fall back to full_start if truncated
                if seed_pos + period > n:
                    seed_pos = full_start

                motif_arr = text_arr[seed_pos:seed_pos + period]
                motif = motif_arr.tobytes().decode('ascii', errors='replace')

                # Anchor-based boundary verification (Cython-accelerated)
                true_start, true_end = anchor_scan_boundaries(
                    text_arr, seed_pos, period, n,
                    anchor_match_pct, scan_bw_periods, scan_fw_periods,
                )

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
