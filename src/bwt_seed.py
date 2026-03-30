"""Shared BWT k-mer seeding core for tandem repeat detection.

Used by both Tier 2 and Tier 3.  The algorithm is:
  1. Sample k-mers from the text at a configurable stride.
  2. Use FM-index backward_search / locate_positions to find all occurrences.
  3. Detect periodic runs in the position arrays (arithmetic progressions).
  4. Extend seed positions with mismatch tolerance.
  5. Return raw candidate regions for tier-specific post-processing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np

from .accelerators import extend_with_mismatches, find_periodic_runs, find_tandem_runs
from .bwt_core import BWTCore


@dataclass
class SeedCandidate:
    """A raw repeat candidate from BWT seeding."""
    start: int          # Repeat array start (after extension)
    end: int            # Repeat array end   (after extension)
    period: int         # Detected period
    copies: int         # Number of copies found
    motif: str          # Representative motif string
    seed_pos: int       # Original seed position that generated this candidate


def bwt_kmer_seed_scan(
    bwt: BWTCore,
    min_period: int,
    max_period: int,
    kmer_size: int = 16,
    stride: int = 50,
    min_copies: int = 2,
    allowed_mismatch_rate: float = 0.20,
    tolerance_ratio: float = 0.03,
    max_occurrences: int = 5000,
    covered_mask: Optional[np.ndarray] = None,
    show_progress: bool = False,
    label: str = "",
) -> List[SeedCandidate]:
    """BWT k-mer seeding scan for tandem repeat candidates.

    Samples k-mers from the text at `stride` intervals, uses FM-index to
    find all occurrences, detects periodic runs in the occurrence positions,
    and extends seed hits with mismatch tolerance.

    Parameters
    ----------
    bwt : BWTCore
        The BWT/FM-index built on the sequence.
    min_period, max_period : int
        Period range to search for.
    kmer_size : int
        Length of k-mers to sample (default 16).  Should be shorter than
        min_period so that each repeat copy contains at least one full k-mer.
    stride : int
        Distance between consecutive k-mer samples (default 50).
    min_copies : int
        Minimum number of tandem copies required.
    allowed_mismatch_rate : float
        Mismatch tolerance for extension (0.0-0.5).
    tolerance_ratio : float
        Tolerance for period jitter in periodic run detection (default 3%).
    max_occurrences : int
        Skip k-mers with more occurrences than this (likely low-complexity).
    covered_mask : np.ndarray or None
        Boolean mask of positions already found by a previous tier.
        Positions where covered_mask[i] is True are skipped as seed origins.
    show_progress : bool
        Whether to print progress.
    label : str
        Label prefix for progress messages.

    Returns
    -------
    list[SeedCandidate]
        Raw repeat candidates.  Callers perform tier-specific post-processing
        (primitive period reduction, HOR detection, DP refinement, etc.).
    """
    text_arr = bwt.text_arr
    n = int(text_arr.size)

    # Exclude sentinel
    if n > 0 and text_arr[n - 1] == 36:   # '$'
        n -= 1

    if n < min_period * min_copies:
        return []

    # Clamp kmer_size to be at most min_period (otherwise k-mer spans >1 copy)
    effective_kmer = min(kmer_size, min_period)
    if effective_kmer < 6:
        effective_kmer = min(6, min_period)

    candidates: List[SeedCandidate] = []
    seen_regions: Set[Tuple[int, int]] = set()   # (start//bucket, period//bucket)
    seen_kmers: Set[str] = set()
    bwt_queries = 0

    i = 0
    while i < n - effective_kmer:
        # Skip positions covered by a previous tier
        if covered_mask is not None and covered_mask[i]:
            i += stride
            continue

        # Extract k-mer
        kmer_arr = text_arr[i:i + effective_kmer]
        kmer_str = kmer_arr.tobytes().decode('ascii', errors='replace')

        # Skip non-DNA or already-tried k-mers
        if kmer_str in seen_kmers:
            i += stride
            continue
        if not all(c in 'ACGT' for c in kmer_str):
            i += stride
            continue
        seen_kmers.add(kmer_str)

        # --- FM-index lookup ---
        bwt_queries += 1
        sp, ep = bwt.backward_search(kmer_str)
        if sp == -1:
            i += stride
            continue

        occ_count = ep - sp + 1
        if occ_count < min_copies or occ_count > max_occurrences:
            i += stride
            continue

        positions = bwt.locate_positions(kmer_str)
        if len(positions) < min_copies:
            i += stride
            continue

        # --- Find periodic runs in occurrence positions ---
        pos_arr = np.array(sorted(positions), dtype=np.int64)

        # Use find_periodic_runs (allows tolerance in spacing)
        patterns = find_periodic_runs(
            pos_arr, min_period, max_period, min_copies, tolerance_ratio
        )

        for run_start, run_end, period in patterns:
            run_start = int(run_start)
            run_end = int(run_end)
            period = int(period)

            # Deduplicate by coarse region key
            region_key = (run_start // max(period, 1), period)
            if region_key in seen_regions:
                continue

            # Skip if heavily overlapping with covered regions
            if covered_mask is not None:
                covered_count = int(np.sum(
                    covered_mask[run_start:min(run_end + period, n)]
                ))
                span = run_end + period - run_start
                if span > 0 and covered_count > span * 0.5:
                    continue

            # --- Extend with mismatch tolerance ---
            res = extend_with_mismatches(
                text_arr, run_start, period, n, allowed_mismatch_rate
            )

            if res is not None:
                arr_start, arr_end, copies, full_start, full_end = res
            else:
                # Fallback: use raw run boundaries
                full_start = run_start
                # run_end from find_periodic_runs is the start of the LAST k-mer
                full_end = run_end + period
                copies = max(1, (full_end - full_start) // period)

            if copies < min_copies:
                continue

            # Extract motif from the confirmed region
            motif_start = max(0, full_start)
            motif_arr = text_arr[motif_start:motif_start + period]
            motif_str = motif_arr.tobytes().decode('ascii', errors='replace')

            candidates.append(SeedCandidate(
                start=full_start,
                end=full_end,
                period=period,
                copies=copies,
                motif=motif_str,
                seed_pos=i,
            ))
            seen_regions.add(region_key)

            # Mark this region so we don't re-seed into it
            if covered_mask is not None:
                covered_mask[full_start:min(full_end, n)] = True

        i += stride

    if show_progress:
        print(
            f"  [{label}] BWT k-mer seeding: {len(candidates)} candidates from "
            f"{bwt_queries} FM-index queries (kmer={effective_kmer}, stride={stride})",
            flush=True,
        )

    return candidates
