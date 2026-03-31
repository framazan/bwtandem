import math  # Standard library for math functions (log10, etc.)
import numpy as np  # NumPy for array operations and numerical computation
from typing import List, Tuple, Set, Optional  # typing module for type hints
from .models import TandemRepeat  # Data class for tandem repeat results
from .motif_utils import MotifUtils  # Motif analysis utilities (alignment, consensus, TRF statistics, etc.)
from .bwt_core import BWTCore  # FM-index core module (BWT, backward search, etc.)
from .accelerators import anchor_scan_boundaries  # Cython-accelerated anchor-based boundary verification
from .bwt_seed import bwt_kmer_seed_scan  # Shared BWT k-mer seeding pipeline


def compute_adaptive_params(
    seq_len: int,        # Input sequence length (bp)
    gc_content: float,   # GC content (range 0 to 1)
    coverage_ratio: float,  # Fraction of sequence already covered by other tiers
    min_period: int,     # Minimum repeat unit length
    max_period: int,     # Maximum repeat unit length
    preset: str = "balanced",  # Speed/sensitivity preset ("fast", "balanced", "sensitive")
) -> dict:
    """Compute adaptive Tier 3 parameters based on input characteristics."""
    # Speed weight per preset: fast=faster (lower sensitivity), sensitive=slower (higher sensitivity)
    speed_weights = {"fast": 0.8, "balanced": 0.5, "sensitive": 0.2}
    speed_weight = speed_weights.get(preset, 0.5)  # Extract weight for the preset (default 0.5 if not found)
    speed_factor = speed_weight / 0.5  # Relative speed multiplier compared to balanced (0.5)

    # Compute base parameters using continuous functions (log-scaled by sequence length)
    safe_seq = max(seq_len, 1)  # Lower bound to prevent division by zero

    base_kmer = int(10 + 6 * math.log10(max(safe_seq / 1e5, 1)))  # k-mer size: larger k-mers for longer sequences
    base_stride = int(safe_seq / 40000)  # Sampling stride: proportional to sequence length
    base_max_occ = int(safe_seq / 30000)  # Max allowed k-mer occurrences: filters out low-complexity high-frequency k-mers
    base_scan_bw = int(50 * safe_seq / 1e8)  # Backward scan range (in repeat units)
    base_scan_fw = int(600 * safe_seq / 1e8)  # Forward scan range (in repeat units)

    # Accuracy parameters: depend only on sequence characteristics, not preset
    allowed_mismatch_rate = 0.15 + 0.10 * abs(gc_content - 0.5)  # Higher mismatch tolerance for extreme GC content
    tolerance_ratio = 0.02 + 0.02 * (max_period / 100000)  # Wider period tolerance for larger max periods
    anchor_match_pct = 0.70 + 0.10 * (1 - coverage_ratio)  # Relax anchor matching threshold when coverage is low

    # Apply preset multiplier to speed parameters
    kmer_size = int(base_kmer + (speed_factor - 1) * 2)  # Fast mode increases k-mer size (fewer hits)
    stride = int(base_stride * speed_factor)  # Fast mode uses larger stride for sparser sampling
    max_occurrences = int(base_max_occ / speed_factor)  # Fast mode allows fewer high-frequency k-mers
    scan_backward = int(base_scan_bw / speed_factor)  # Fast mode scans shorter backward range
    scan_forward = int(base_scan_fw / speed_factor)  # Fast mode scans shorter forward range

    # When coverage exceeds 50%, reduce stride to scan uncovered regions more finely
    if coverage_ratio > 0.5:
        stride = int(stride * (1 - 0.5 * coverage_ratio))  # Higher coverage leads to smaller stride

    # Strategy adjustments by sequence size range
    if safe_seq > 100_000_000:  # Over 100 Mbp: large chromosome mode
        stride = max(stride, 150)  # Ensure minimum stride to avoid excessive slowdown
        kmer_size = max(kmer_size, 20)  # Use longer k-mers for specificity on large sequences
        max_occurrences = min(max_occurrences, 500)  # Tighten upper bound to filter low-complexity repeats
    elif safe_seq < 100_000:  # Under 100 kbp: micro mode (short sequences)
        stride = max(stride, 20)  # Maintain minimum sampling stride even for short sequences
        kmer_size = max(kmer_size, 12)  # Ensure minimum k-mer size for short sequences

    # Clamp all parameters to allowed ranges
    kmer_size = max(12, min(28, kmer_size))  # k-mer size: restricted to 12-28 bp
    stride = max(20, min(300, stride))  # Sampling stride: restricted to 20-300
    allowed_mismatch_rate = max(0.15, min(0.20, allowed_mismatch_rate))  # Mismatch rate: 15%-20%
    tolerance_ratio = max(0.02, min(0.04, tolerance_ratio))  # Period tolerance: 2%-4%
    max_occurrences = max(200, min(1500, max_occurrences))  # Max occurrences: 200-1500
    anchor_match_pct = max(0.70, min(0.80, anchor_match_pct))  # Anchor match ratio: 70%-80%
    scan_backward = max(20, min(80, scan_backward))  # Backward scan: 20-80 units
    scan_forward = max(200, min(800, scan_forward))  # Forward scan: 200-800 units

    # Return final parameters as a dictionary
    return {
        "kmer_size": kmer_size,              # k-mer length for FM-index lookup
        "stride": stride,                    # k-mer sampling stride
        "allowed_mismatch_rate": allowed_mismatch_rate,  # Allowed mismatch rate during extension
        "tolerance_ratio": tolerance_ratio,  # Period detection tolerance ratio
        "max_occurrences": max_occurrences,  # Max k-mer occurrences (low-complexity filter)
        "anchor_match_pct": anchor_match_pct,  # Anchor-based boundary verification match threshold
        "scan_backward": scan_backward,      # Anchor backward scan range (in repeat units)
        "scan_forward": scan_forward,        # Anchor forward scan range (in repeat units)
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
        self.bwt = bwt_core          # Store FM-index object (used for subsequent searches)
        self.min_length = min_length  # Minimum repeat unit length to detect (bp)
        self.max_length = max_length  # Maximum repeat unit length to detect (bp)
        self.min_copies = min_copies  # Minimum number of copies required to qualify as a repeat
        self.mode = mode             # Speed/sensitivity preset string

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
        text_arr = self.bwt.text_arr  # Original sequence used for BWT (numpy uint8 array)
        n = text_arr.size             # Total sequence length (including sentinel '$')

        # Sequence too short for detection; return empty results
        if n < self.min_length * 2:
            return []

        # Create boolean mask marking regions already found by previous tiers
        mask = np.zeros(n, dtype=bool)  # Initially all positions are uncovered
        for start, end in tier1_seen:
            mask[start:min(end, n)] = True  # Mark regions found by Tier 1
        for start, end in tier2_seen:
            mask[start:min(end, n)] = True  # Mark regions found by Tier 2

        # Compute adaptive parameters based on sequence characteristics
        gc_content = float(np.mean((text_arr == ord('G')) | (text_arr == ord('C'))))  # Compute GC content
        coverage_ratio = float(np.mean(mask))  # Compute fraction of already-covered regions
        params = compute_adaptive_params(
            seq_len=n,
            gc_content=gc_content,
            coverage_ratio=coverage_ratio,
            min_period=self.min_length,
            max_period=self.max_length,
            preset=self.mode,
        )  # Parameter dictionary reflecting sequence length, GC content, coverage, etc.

        anchor_match_pct = params["anchor_match_pct"]  # Extract anchor-based boundary verification match threshold
        scan_bw_periods = params["scan_backward"]      # Extract backward anchor scan range
        scan_fw_periods = params["scan_forward"]       # Extract forward anchor scan range

        # ===== Phase A: Run shared BWT k-mer seeding pipeline =====
        seed_candidates = bwt_kmer_seed_scan(
            bwt=self.bwt,
            min_period=self.min_length,       # Minimum repeat unit length
            max_period=self.max_length,       # Maximum repeat unit length
            kmer_size=params["kmer_size"],    # k-mer size for FM-index lookup
            stride=params["stride"],          # k-mer sampling stride
            min_copies=int(self.min_copies),  # Minimum copies (cast to int)
            allowed_mismatch_rate=params["allowed_mismatch_rate"],  # Mismatch tolerance rate
            tolerance_ratio=params["tolerance_ratio"],              # Period tolerance ratio
            max_occurrences=params["max_occurrences"],              # Max k-mer occurrences
            covered_mask=mask,                # Mask of already-detected regions (skip seeding positions)
            show_progress=False,              # Disable progress output
            label=f"{chromosome} Tier3",      # Label for progress messages
        )  # Returns: list of SeedCandidate objects (raw repeat candidates)

        # ===== Tier 3 후처리: 후보를 TandemRepeat 객체로 변환 =====
        repeats = []          # 최종 결과 리스트
        seen_regions = set()  # 중복 후보 제거를 위한 영역 키 집합

        for cand in seed_candidates:  # 각 시드 후보에 대해 후처리 수행
            region_key = (cand.start // max(cand.period, 1), cand.period)  # 반복 영역 식별 키 (중복 제거용)
            if region_key in seen_regions:  # 이미 처리된 영역이면 건너뜀
                continue

            period = cand.period      # 탐지된 반복 단위 길이
            copies = cand.copies      # 탐지된 복사 수
            full_start = cand.start   # 확장 후 반복 배열 시작 위치
            full_end = cand.end       # 확장 후 반복 배열 끝 위치

            # 초장대 반복 (복사 수 >100 또는 길이 >10kb): 앵커 기반 경계 검증 사용
            # 전체 DP 정렬은 비용이 크므로 앵커 스캔으로 대체
            if copies > 100 or (full_end - full_start) > 10000:
                seed_pos = cand.seed_pos  # 이 후보를 생성한 원본 시드 위치
                # 시드 위치가 서열 끝을 벗어나면 full_start로 폴백
                if seed_pos + period > n:
                    seed_pos = full_start

                motif_arr = text_arr[seed_pos:seed_pos + period]  # 시드 위치에서 모티프 추출
                motif = motif_arr.tobytes().decode('ascii', errors='replace')  # 바이트 배열을 문자열로 변환

                # 앵커 기반 경계 검증: Cython 가속으로 실제 반복 시작/끝 위치 정밀화
                true_start, true_end = anchor_scan_boundaries(
                    text_arr, seed_pos, period, n,
                    anchor_match_pct, scan_bw_periods, scan_fw_periods,
                )  # 앵커 매칭 비율과 스캔 범위를 기반으로 경계 재계산

                true_copies = (true_end - true_start) / period  # 실제 복사 수 재계산

                if true_copies >= self.min_copies:  # 최소 복사 수 조건을 만족하는 경우만 결과로 처리
                    max_consensus_copies = min(int(true_copies), 20)  # 컨센서스 계산에 사용할 최대 복사 수 (효율을 위해 20으로 제한)
                    consensus_arr, mm_rate, max_mm = MotifUtils.build_consensus_motif_array(
                        text_arr, true_start, period, max_consensus_copies
                    )  # 샘플링된 복사들로 컨센서스 모티프와 미스매치 통계 계산
                    consensus_motif = consensus_arr.tobytes().decode('ascii', errors='replace') if consensus_arr.size > 0 else motif
                    # 컨센서스 배열이 비어 있으면 원본 모티프로 폴백

                    (percent_matches, percent_indels, score, composition,
                     entropy, actual_sequence) = MotifUtils.calculate_trf_statistics(
                        text_arr, true_start, true_end, consensus_motif, int(true_copies), mm_rate
                    )  # TRF 호환 통계 계산: 매칭률, 삽입결실률, 점수, 염기 조성, 엔트로피, 실제 서열

                    # TandemRepeat 객체 생성 (앵커 기반 경계 사용)
                    repeat = TandemRepeat(
                        chrom=chromosome,           # 염색체 이름
                        start=true_start,           # 실제 반복 시작 위치 (0-based)
                        end=true_end,               # 실제 반복 끝 위치 (0-based)
                        motif=motif,                # 원본 시드에서 추출한 모티프
                        copies=float(true_copies),  # 실제 복사 수 (float)
                        length=true_end - true_start,  # 반복 배열 총 길이
                        tier=3,                     # 이 결과가 Tier 3에서 생성됨을 표시
                        confidence=max(0.5, 1.0 - mm_rate),  # 신뢰도: 미스매치율이 낮을수록 높음 (최소 0.5)
                        consensus_motif=consensus_motif,      # 컨센서스 기반 모티프
                        mismatch_rate=mm_rate,                # 평균 미스매치율
                        max_mismatches_per_copy=max_mm,       # 복사당 최대 미스매치 수
                        n_copies_evaluated=max_consensus_copies,  # 컨센서스 계산에 사용된 복사 수
                        strand='+',                           # 정방향 가닥 (Tier3는 항상 '+')
                        percent_matches=percent_matches,      # TRF 통계: 매칭 비율
                        percent_indels=percent_indels,        # TRF 통계: 삽입결실 비율
                        score=score,                          # TRF 통계: 점수
                        composition=composition,              # TRF 통계: 염기 조성
                        entropy=entropy,                      # TRF 통계: 서열 엔트로피
                        actual_sequence=actual_sequence[:500] if len(actual_sequence) > 500 else actual_sequence,
                        # 실제 서열: 출력 크기 제한을 위해 500bp로 절단
                        variations=None  # Tier3는 변이 정보를 기록하지 않음
                    )
                else:
                    repeat = None  # 최소 복사 수 미달: 결과 없음
            else:
                # 상대적으로 짧은 반복 (<100복사 또는 <10kb): 전체 DP 정렬로 정밀화
                motif_arr = text_arr[full_start:full_start + period]  # 반복 시작 위치에서 모티프 추출
                motif = motif_arr.tobytes().decode('ascii', errors='replace')  # 바이트 배열을 문자열로 변환

                refined = MotifUtils.refine_repeat(
                    self.bwt.text,        # 원본 텍스트 (문자열)
                    full_start,           # 반복 배열 시작 위치
                    full_end,             # 반복 배열 끝 위치
                    motif,                # 초기 모티프
                    mismatch_fraction=0.2,  # 허용 미스매치 비율 (20%)
                    indel_fraction=0.1,   # 허용 삽입결실 비율 (10%)
                    min_copies=int(self.min_copies)  # 최소 복사 수
                )  # DP 정렬 기반 반복 서열 정밀화 (원시 주기 감소 포함)

                if refined:
                    repeat = MotifUtils.refined_to_repeat(chromosome, refined, tier=3, text_arr=text_arr)
                    # 정밀화 결과를 TandemRepeat 객체로 변환
                else:
                    repeat = None  # DP 정밀화 실패: 결과 없음

            if repeat:
                # 기존 결과와의 포함 관계 확인 (완전히 포함된 경우 새 결과 무시)
                is_new = True  # 기본적으로 새 결과로 간주
                for r in repeats:
                    if r.start <= repeat.start and r.end >= repeat.end:
                        is_new = False  # 기존 결과가 현재 반복을 완전히 포함하면 중복으로 처리
                        break

                if is_new:
                    repeats.append(repeat)             # 최종 결과 리스트에 추가
                    seen_regions.add(region_key)       # 해당 영역 키를 처리 완료로 등록
                    mask[repeat.start:min(repeat.end, n)] = True  # 발견된 영역을 마스크에 표시 (후속 시딩 스킵)

        return repeats  # 발견된 모든 장대 반복 서열 결과 반환
