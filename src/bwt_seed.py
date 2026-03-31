"""Shared BWT k-mer seeding core for tandem repeat detection.

Used by both Tier 2 and Tier 3.  The algorithm is:
  1. Sample k-mers from the text at a configurable stride.
  2. Use FM-index backward_search / locate_positions to find all occurrences.
  3. Detect periodic runs in the position arrays (arithmetic progressions).
  4. Extend seed positions with mismatch tolerance.
  5. Return raw candidate regions for tier-specific post-processing.
"""
from __future__ import annotations  # Allow string-form type hints for Python 3.9 and below

from dataclasses import dataclass   # Decorator for defining immutable data containers
from typing import List, Optional, Set, Tuple  # Generic types for type hints

import numpy as np  # NumPy for array operations and numerical computation

from .accelerators import extend_with_mismatches, find_periodic_runs, find_tandem_runs
# Cython/Python accelerated functions: mismatch-tolerant extension, periodic run detection, tandem run detection
from .bwt_core import BWTCore  # FM-index core module (BWT, backward_search, locate_positions, etc.)


@dataclass
class SeedCandidate:
    """A raw repeat candidate from BWT seeding."""
    start: int      # Start position of the repeat array after extension (0-based)
    end: int        # End position of the repeat array after extension (0-based)
    period: int     # Detected repeat unit length (bp)
    copies: int     # Detected number of copies
    motif: str      # Representative motif string (extracted from full_start position)
    seed_pos: int   # Original seed position that generated this candidate (value of i)


def bwt_kmer_seed_scan(
    bwt: BWTCore,                        # FM-index object
    min_period: int,                     # Minimum repeat unit length to detect
    max_period: int,                     # Maximum repeat unit length to detect
    kmer_size: int = 16,                 # Length of k-mers to sample (default 16 bp)
    stride: int = 50,                    # K-mer sampling interval (default 50 bp)
    min_copies: int = 2,                 # Minimum number of copies to qualify as a repeat
    allowed_mismatch_rate: float = 0.20, # Allowed mismatch rate during extension (0.0-0.5)
    tolerance_ratio: float = 0.03,       # Tolerance ratio for period jitter in periodic run detection (default 3%)
    max_occurrences: int = 5000,         # Maximum k-mer occurrence count (skip if exceeded, likely low-complexity)
    covered_mask: Optional[np.ndarray] = None,  # Boolean mask of positions already found by previous tiers
    show_progress: bool = False,         # Whether to print progress
    label: str = "",                     # Label string for progress messages
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
    text_arr = bwt.text_arr  # Original sequence used for BWT (numpy uint8 array)
    n = int(text_arr.size)   # Total sequence length (may include sentinel '$')

    # Exclude the sentinel character ('$', ASCII 36) used for BWT construction
    if n > 0 and text_arr[n - 1] == 36:   # '$'
        n -= 1  # Use actual sequence length excluding sentinel

    # Return immediately if sequence is too short to contain repeats meeting minimum copy count
    if n < min_period * min_copies:
        return []

    # Clamp k-mer size if larger than min_period (k-mer must fit within one copy)
    effective_kmer = min(kmer_size, min_period)  # Limit to not exceed repeat unit length
    if effective_kmer < 6:
        effective_kmer = min(6, min_period)  # Minimum 6 bp: shorter k-mers cause explosive FM-index hits

    candidates: List[SeedCandidate] = []               # List of detected repeat candidates
    seen_regions: Set[Tuple[int, int]] = set()         # Set of (start//bucket, period) keys for duplicate removal
    seen_kmers: Set[str] = set()                       # Set of already-queried k-mers (avoid re-querying)
    bwt_queries = 0  # FM-index query counter (for progress reporting)

    i = 0  # Current sampling position (incremented by stride)
    while i < n - effective_kmer:  # Iterate until k-mer would be truncated at sequence end
        # Skip positions already found by previous tiers
        if covered_mask is not None and covered_mask[i]:
            i += stride  # Position already covered; move to next sample position
            continue

        # Extract k-mer at current position
        kmer_arr = text_arr[i:i + effective_kmer]  # Sequence slice for the k-mer
        kmer_str = kmer_arr.tobytes().decode('ascii', errors='replace')  # Convert byte array to string

        # Skip if k-mer was already queried or contains non-DNA bases
        if kmer_str in seen_kmers:  # Check cache to avoid duplicate queries
            i += stride
            continue
        if not all(c in 'ACGT' for c in kmer_str):  # Skip if contains N, lowercase, or other non-DNA characters
            i += stride
            continue
        seen_kmers.add(kmer_str)  # Register queried k-mer in cache

        # --- Query all occurrence positions of k-mer via FM-index (BWT backward search) ---
        bwt_queries += 1  # Increment FM-index query counter
        sp, ep = bwt.backward_search(kmer_str)  # Returns suffix array range [sp, ep]
        if sp == -1:  # Not found (should not happen in theory, but defensive check)
            i += stride
            continue

        occ_count = ep - sp + 1  # Calculate total occurrence count of the k-mer
        if occ_count < min_copies or occ_count > max_occurrences:
            # Skip if too few occurrences (no repeat) or too many (low-complexity sequence)
            i += stride
            continue

        positions = bwt.locate_positions(kmer_str)  # Return all positions where k-mer occurs
        if len(positions) < min_copies:  # Skip if actual position count is below minimum copies
            i += stride
            continue

        # --- Detect arithmetic progressions (periodic runs) in occurrence position array ---
        pos_arr = np.array(sorted(positions), dtype=np.int64)  # Convert positions to sorted int64 array
        # (Sorting is required for accurate arithmetic progression detection)

        # find_periodic_runs: detect evenly-spaced groups within tolerance_ratio in position array
        patterns = find_periodic_runs(
            pos_arr, min_period, max_period, min_copies, tolerance_ratio
        )  # Returns: [(run_start, run_end, period), ...] -- list of periodic k-mer occurrence runs

        for run_start, run_end, period in patterns:  # 각 주기적 런에 대해 처리
            run_start = int(run_start)  # numpy 타입을 파이썬 int로 변환
            run_end = int(run_end)      # numpy 타입을 파이썬 int로 변환
            period = int(period)        # numpy 타입을 파이썬 int로 변환

            # 동일한 영역의 중복 후보 제거: 시작 위치와 주기의 조합으로 키 생성
            region_key = (run_start // max(period, 1), period)  # 반복 구간 식별 키
            if region_key in seen_regions:  # 이미 처리된 영역이면 건너뜀
                continue

            # 이미 커버된 영역과 50% 이상 겹치는 경우 스킵 (중복 탐지 방지)
            if covered_mask is not None:
                covered_count = int(np.sum(
                    covered_mask[run_start:min(run_end + period, n)]
                ))  # 해당 구간 내 이미 커버된 위치 수 계산
                span = run_end + period - run_start  # 런의 전체 길이 계산 (마지막 k-mer 포함)
                if span > 0 and covered_count > span * 0.5:
                    continue  # 50% 초과 커버 시 건너뜀

            # --- 미스매치 허용 확장: 반복 배열의 실제 경계를 양방향으로 확장 ---
            res = extend_with_mismatches(
                text_arr, run_start, period, n, allowed_mismatch_rate
            )  # 반환: (arr_start, arr_end, copies, full_start, full_end) 또는 None

            if res is not None:
                arr_start, arr_end, copies, full_start, full_end = res
                # arr_start/arr_end: 정렬 기준 경계, full_start/full_end: 실제 확장된 경계
            else:
                # 확장 실패 시 폴백: k-mer 런의 원시 경계 사용
                full_start = run_start
                # run_end는 마지막 k-mer의 시작 위치이므로 period를 더해 끝 위치 계산
                full_end = run_end + period
                copies = max(1, (full_end - full_start) // period)  # 복사 수 추정

            if copies < min_copies:  # 최소 복사 수 미달이면 후보로 등록하지 않음
                continue

            # 확인된 반복 영역에서 모티프 문자열 추출
            motif_start = max(0, full_start)  # 음수 인덱스 방지를 위한 클램프
            motif_arr = text_arr[motif_start:motif_start + period]  # 한 복사에 해당하는 서열 슬라이스
            motif_str = motif_arr.tobytes().decode('ascii', errors='replace')  # 바이트 배열을 문자열로 변환

            # SeedCandidate 객체 생성 및 후보 목록에 추가
            candidates.append(SeedCandidate(
                start=full_start,   # 확장된 반복 배열 시작 위치
                end=full_end,       # 확장된 반복 배열 끝 위치
                period=period,      # 탐지된 반복 단위 길이
                copies=copies,      # 탐지된 복사 수
                motif=motif_str,    # 대표 모티프 문자열
                seed_pos=i,         # 이 후보를 생성한 원본 시드 위치
            ))
            seen_regions.add(region_key)  # 해당 영역을 처리 완료로 등록

            # 발견된 영역을 마스크에 표시: 이후 같은 구간에서 재시딩하지 않도록 함
            if covered_mask is not None:
                covered_mask[full_start:min(full_end, n)] = True  # 해당 구간을 커버됨으로 표시

        i += stride  # 다음 샘플 위치로 이동

    # 진행 상황 출력 (show_progress가 True일 때만)
    if show_progress:
        print(
            f"  [{label}] BWT k-mer seeding: {len(candidates)} candidates from "
            f"{bwt_queries} FM-index queries (kmer={effective_kmer}, stride={stride})",
            flush=True,
        )  # 후보 수, FM-인덱스 조회 횟수, 실효 k-mer 크기, 샘플링 간격 출력

    return candidates  # 탐지된 모든 원시 반복 후보 목록 반환
