"""Shared BWT k-mer seeding core for tandem repeat detection.

Used by both Tier 2 and Tier 3.  The algorithm is:
  1. Sample k-mers from the text at a configurable stride.
  2. Use FM-index backward_search / locate_positions to find all occurrences.
  3. Detect periodic runs in the position arrays (arithmetic progressions).
  4. Extend seed positions with mismatch tolerance.
  5. Return raw candidate regions for tier-specific post-processing.
"""
from __future__ import annotations  # 파이썬 3.9 이하에서도 타입 힌트 문자열 형식 사용 허용

from dataclasses import dataclass   # 불변 데이터 컨테이너 정의를 위한 데코레이터
from typing import List, Optional, Set, Tuple  # 타입 힌트용 제네릭 타입들

import numpy as np  # 배열 연산 및 수치 계산을 위한 NumPy

from .accelerators import extend_with_mismatches, find_periodic_runs, find_tandem_runs
# Cython/Python 가속 함수: 미스매치 허용 확장, 주기적 런 탐지, 탄뎀 런 탐지
from .bwt_core import BWTCore  # FM-인덱스 핵심 모듈 (BWT, backward_search, locate_positions 등)


@dataclass
class SeedCandidate:
    """A raw repeat candidate from BWT seeding."""
    start: int      # 확장 후 반복 배열의 시작 위치 (0-based)
    end: int        # 확장 후 반복 배열의 끝 위치 (0-based)
    period: int     # 탐지된 반복 단위 길이 (bp)
    copies: int     # 탐지된 복사 수
    motif: str      # 대표 모티프 문자열 (full_start 위치에서 추출)
    seed_pos: int   # 이 후보를 생성한 원본 시드 위치 (i 값)


def bwt_kmer_seed_scan(
    bwt: BWTCore,                        # FM-인덱스 객체
    min_period: int,                     # 탐지할 최소 반복 단위 길이
    max_period: int,                     # 탐지할 최대 반복 단위 길이
    kmer_size: int = 16,                 # 샘플링할 k-mer 길이 (기본값 16bp)
    stride: int = 50,                    # k-mer 샘플링 간격 (기본값 50bp)
    min_copies: int = 2,                 # 반복으로 인정하기 위한 최소 복사 수
    allowed_mismatch_rate: float = 0.20, # 확장 시 허용 미스매치 비율 (0.0~0.5)
    tolerance_ratio: float = 0.03,       # 주기 탐지 시 간격 오차 허용 비율 (기본 3%)
    max_occurrences: int = 5000,         # k-mer 최대 발생 횟수 (초과 시 저복잡 서열로 간주하고 스킵)
    covered_mask: Optional[np.ndarray] = None,  # 이전 티어에서 이미 발견된 위치 마스크
    show_progress: bool = False,         # 진행 상황 출력 여부
    label: str = "",                     # 진행 메시지에 붙일 레이블 문자열
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
    text_arr = bwt.text_arr  # BWT에 사용된 원본 서열 (numpy uint8 배열)
    n = int(text_arr.size)   # 서열 전체 길이 (센티넬 '$' 포함 가능)

    # BWT 구성에 사용된 센티넬 문자('$', ASCII 36) 제외: 마지막 문자가 '$'이면 n 감소
    if n > 0 and text_arr[n - 1] == 36:   # '$'
        n -= 1  # 센티넬을 제외하고 실제 서열 길이만 사용

    # 서열이 너무 짧아 최소 복사 수를 충족하는 반복을 포함할 수 없으면 즉시 반환
    if n < min_period * min_copies:
        return []

    # k-mer 크기가 min_period보다 크면 한 복사 안에 k-mer가 들어가지 않으므로 clamp
    effective_kmer = min(kmer_size, min_period)  # 반복 단위 길이를 초과하지 않도록 제한
    if effective_kmer < 6:
        effective_kmer = min(6, min_period)  # 최소 6bp: 너무 짧으면 FM-인덱스 히트가 폭발적으로 증가함

    candidates: List[SeedCandidate] = []               # 탐지된 반복 후보 목록
    seen_regions: Set[Tuple[int, int]] = set()         # 중복 후보 제거용 (start//bucket, period) 키 집합
    seen_kmers: Set[str] = set()                       # 이미 FM-인덱스에서 조회한 k-mer 집합 (재조회 방지)
    bwt_queries = 0  # FM-인덱스 조회 횟수 카운터 (진행 상황 보고용)

    i = 0  # 현재 샘플링 위치 (stride씩 증가)
    while i < n - effective_kmer:  # 서열 끝에서 k-mer가 잘리지 않는 범위까지 반복
        # 이전 티어에서 발견된 위치는 시드로 사용하지 않고 건너뜀
        if covered_mask is not None and covered_mask[i]:
            i += stride  # 해당 위치는 이미 커버됨 → 다음 샘플 위치로 이동
            continue

        # 현재 위치에서 k-mer 추출
        kmer_arr = text_arr[i:i + effective_kmer]  # k-mer에 해당하는 서열 슬라이스
        kmer_str = kmer_arr.tobytes().decode('ascii', errors='replace')  # 바이트 배열을 문자열로 변환

        # 이미 조회한 k-mer이거나 비-DNA 염기 포함 시 건너뜀
        if kmer_str in seen_kmers:  # 동일한 k-mer를 중복 조회하지 않도록 캐시 확인
            i += stride
            continue
        if not all(c in 'ACGT' for c in kmer_str):  # N, 소문자 등 비-DNA 문자 포함 시 스킵
            i += stride
            continue
        seen_kmers.add(kmer_str)  # 조회한 k-mer를 캐시에 등록

        # --- FM-인덱스(BWT backward search)로 k-mer의 모든 발생 위치 조회 ---
        bwt_queries += 1  # FM-인덱스 조회 횟수 증가
        sp, ep = bwt.backward_search(kmer_str)  # suffix array 범위 [sp, ep] 반환
        if sp == -1:  # 발견 안 됨 (이론상 발생하지 않아야 하지만 방어 코드)
            i += stride
            continue

        occ_count = ep - sp + 1  # k-mer의 총 발생 횟수 계산
        if occ_count < min_copies or occ_count > max_occurrences:
            # 발생 횟수가 너무 적으면(반복 없음) 또는 너무 많으면(저복잡 서열) 스킵
            i += stride
            continue

        positions = bwt.locate_positions(kmer_str)  # k-mer가 발생한 모든 위치 목록 반환
        if len(positions) < min_copies:  # 실제 위치 수가 최소 복사 수 미달이면 스킵
            i += stride
            continue

        # --- 발생 위치 배열에서 등차수열(주기적 런) 탐지 ---
        pos_arr = np.array(sorted(positions), dtype=np.int64)  # 위치를 정렬된 int64 배열로 변환
        # (정렬해야 등차수열 패턴 탐지가 정확히 작동함)

        # find_periodic_runs: 위치 배열에서 허용 오차(tolerance_ratio) 내의 등간격 그룹 탐지
        patterns = find_periodic_runs(
            pos_arr, min_period, max_period, min_copies, tolerance_ratio
        )  # 반환: [(run_start, run_end, period), ...] — 주기적 k-mer 발생 런 목록

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
