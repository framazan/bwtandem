import math  # 수학 함수(log10 등) 사용을 위한 표준 라이브러리
import numpy as np  # 배열 연산 및 수치 계산을 위한 NumPy
from typing import List, Tuple, Set, Optional  # 타입 힌트를 위한 typing 모듈
from .models import TandemRepeat  # 반복 서열 결과를 담는 데이터 클래스
from .motif_utils import MotifUtils  # 모티프 분석 유틸리티 (정렬, 컨센서스, TRF 통계 등)
from .bwt_core import BWTCore  # FM-인덱스 핵심 모듈 (BWT, 역방향 탐색 등)
from .accelerators import anchor_scan_boundaries  # Cython 가속: 앵커 기반 경계 검증
from .bwt_seed import bwt_kmer_seed_scan  # 공유 BWT k-mer 시딩 파이프라인


def compute_adaptive_params(
    seq_len: int,        # 입력 서열 길이 (bp)
    gc_content: float,   # GC 함량 (0~1 범위)
    coverage_ratio: float,  # 이미 다른 티어에서 발견된 영역 비율
    min_period: int,     # 최소 반복 단위 길이
    max_period: int,     # 최대 반복 단위 길이
    preset: str = "balanced",  # 속도/민감도 프리셋 ("fast", "balanced", "sensitive")
) -> dict:
    """Compute adaptive Tier 3 parameters based on input characteristics."""
    # 프리셋별 속도 가중치 정의: fast=빠름(민감도 낮음), sensitive=느림(민감도 높음)
    speed_weights = {"fast": 0.8, "balanced": 0.5, "sensitive": 0.2}
    speed_weight = speed_weights.get(preset, 0.5)  # 프리셋에 해당하는 가중치 추출 (없으면 기본 0.5)
    speed_factor = speed_weight / 0.5  # balanced(0.5) 대비 상대적 속도 배율 계산

    # 연속 함수 기반의 기본 파라미터 계산 (서열 길이에 따라 로그 스케일로 조정)
    safe_seq = max(seq_len, 1)  # 0으로 나누기 방지를 위한 하한값 적용

    base_kmer = int(10 + 6 * math.log10(max(safe_seq / 1e5, 1)))  # k-mer 크기: 서열이 길수록 더 큰 k-mer 사용
    base_stride = int(safe_seq / 40000)  # 샘플링 간격: 서열 길이에 비례
    base_max_occ = int(safe_seq / 30000)  # 최대 허용 k-mer 발생 횟수: 고반복 저복잡 서열 제외용
    base_scan_bw = int(50 * safe_seq / 1e8)  # 역방향 스캔 범위 (반복 단위 수)
    base_scan_fw = int(600 * safe_seq / 1e8)  # 순방향 스캔 범위 (반복 단위 수)

    # 정확도 파라미터: 프리셋에 영향받지 않고 서열 특성에만 의존
    allowed_mismatch_rate = 0.15 + 0.10 * abs(gc_content - 0.5)  # GC가 극단적일수록 미스매치 허용률 증가
    tolerance_ratio = 0.02 + 0.02 * (max_period / 100000)  # 반복 주기 최대값이 클수록 주기 오차 허용 증가
    anchor_match_pct = 0.70 + 0.10 * (1 - coverage_ratio)  # 커버리지가 낮을수록 앵커 매칭 기준 완화

    # 속도 파라미터에 프리셋 배율 적용
    kmer_size = int(base_kmer + (speed_factor - 1) * 2)  # fast 모드는 k-mer 크기 증가 (더 적은 히트)
    stride = int(base_stride * speed_factor)  # fast 모드는 더 큰 간격으로 드물게 샘플링
    max_occurrences = int(base_max_occ / speed_factor)  # fast 모드는 고발생 k-mer 더 적게 허용
    scan_backward = int(base_scan_bw / speed_factor)  # fast 모드는 더 짧게 역방향 스캔
    scan_forward = int(base_scan_fw / speed_factor)  # fast 모드는 더 짧게 순방향 스캔

    # 커버리지가 50% 초과 시 stride를 줄여 미탐지 영역을 더 세밀하게 탐색
    if coverage_ratio > 0.5:
        stride = int(stride * (1 - 0.5 * coverage_ratio))  # 커버리지가 높을수록 stride 감소

    # 서열 크기 구간별 전략 조정
    if safe_seq > 100_000_000:  # 100 Mbp 초과: 대형 염색체 모드
        stride = max(stride, 150)  # 너무 느려지지 않도록 stride 최솟값 보장
        kmer_size = max(kmer_size, 20)  # 대형 서열에서는 더 긴 k-mer로 특이성 확보
        max_occurrences = min(max_occurrences, 500)  # 저복잡 반복을 걸러내기 위해 상한 강화
    elif safe_seq < 100_000:  # 100 kbp 미만: 마이크로 모드 (짧은 서열)
        stride = max(stride, 20)  # 짧은 서열에서도 최소한의 샘플링 간격 유지
        kmer_size = max(kmer_size, 12)  # 짧은 서열에는 최소 k-mer 크기 보장

    # 모든 파라미터를 허용 범위 내로 클램핑
    kmer_size = max(12, min(28, kmer_size))  # k-mer 크기: 12~28 bp 범위로 제한
    stride = max(20, min(300, stride))  # 샘플링 간격: 20~300 범위로 제한
    allowed_mismatch_rate = max(0.15, min(0.20, allowed_mismatch_rate))  # 미스매치율: 15%~20% 범위
    tolerance_ratio = max(0.02, min(0.04, tolerance_ratio))  # 주기 오차율: 2%~4% 범위
    max_occurrences = max(200, min(1500, max_occurrences))  # 최대 발생 횟수: 200~1500 범위
    anchor_match_pct = max(0.70, min(0.80, anchor_match_pct))  # 앵커 매칭 비율: 70%~80% 범위
    scan_backward = max(20, min(80, scan_backward))  # 역방향 스캔: 20~80 단위 범위
    scan_forward = max(200, min(800, scan_forward))  # 순방향 스캔: 200~800 단위 범위

    # 최종 파라미터를 딕셔너리로 반환
    return {
        "kmer_size": kmer_size,              # FM-인덱스 조회에 사용할 k-mer 길이
        "stride": stride,                    # k-mer 샘플링 간격
        "allowed_mismatch_rate": allowed_mismatch_rate,  # 확장 시 허용 미스매치 비율
        "tolerance_ratio": tolerance_ratio,  # 주기 탐지 허용 오차 비율
        "max_occurrences": max_occurrences,  # k-mer 최대 발생 횟수 (저복잡 필터)
        "anchor_match_pct": anchor_match_pct,  # 앵커 기반 경계 검증 매칭 임계값
        "scan_backward": scan_backward,      # 앵커 역방향 스캔 범위 (반복 단위 수)
        "scan_forward": scan_forward,        # 앵커 순방향 스캔 범위 (반복 단위 수)
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
        self.bwt = bwt_core          # FM-인덱스 객체 저장 (이후 탐색에 사용)
        self.min_length = min_length  # 탐지할 최소 반복 단위 길이 (bp)
        self.max_length = max_length  # 탐지할 최대 반복 단위 길이 (bp)
        self.min_copies = min_copies  # 반복으로 인정하기 위한 최소 복사 수
        self.mode = mode             # 속도/민감도 프리셋 문자열

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
        text_arr = self.bwt.text_arr  # BWT에 사용된 원본 서열 (numpy uint8 배열)
        n = text_arr.size             # 서열 전체 길이 (센티넬 '$' 포함)

        # 서열이 너무 짧으면 탐지 불가 → 빈 결과 반환
        if n < self.min_length * 2:
            return []

        # 이전 티어(Tier1, Tier2)가 이미 발견한 영역을 True로 표시하는 불리언 마스크 생성
        mask = np.zeros(n, dtype=bool)  # 초기에는 모든 위치 미탐지 상태
        for start, end in tier1_seen:
            mask[start:min(end, n)] = True  # Tier1이 발견한 구간을 마스크에 표시
        for start, end in tier2_seen:
            mask[start:min(end, n)] = True  # Tier2가 발견한 구간을 마스크에 표시

        # 서열 특성 기반 적응형 파라미터 계산
        gc_content = float(np.mean((text_arr == ord('G')) | (text_arr == ord('C'))))  # GC 함량 계산
        coverage_ratio = float(np.mean(mask))  # 이미 커버된 영역 비율 계산
        params = compute_adaptive_params(
            seq_len=n,
            gc_content=gc_content,
            coverage_ratio=coverage_ratio,
            min_period=self.min_length,
            max_period=self.max_length,
            preset=self.mode,
        )  # 서열 길이·GC함량·커버리지 등을 반영한 파라미터 딕셔너리

        anchor_match_pct = params["anchor_match_pct"]  # 앵커 기반 경계 검증 매칭 임계값 추출
        scan_bw_periods = params["scan_backward"]      # 역방향 앵커 스캔 범위 추출
        scan_fw_periods = params["scan_forward"]       # 순방향 앵커 스캔 범위 추출

        # ===== Phase A: 공유 BWT k-mer 시딩 파이프라인 실행 =====
        seed_candidates = bwt_kmer_seed_scan(
            bwt=self.bwt,
            min_period=self.min_length,       # 최소 반복 단위 길이
            max_period=self.max_length,       # 최대 반복 단위 길이
            kmer_size=params["kmer_size"],    # FM-인덱스 조회에 사용할 k-mer 크기
            stride=params["stride"],          # k-mer 샘플링 간격
            min_copies=int(self.min_copies),  # 최소 복사 수 (정수 변환)
            allowed_mismatch_rate=params["allowed_mismatch_rate"],  # 미스매치 허용률
            tolerance_ratio=params["tolerance_ratio"],              # 주기 오차 허용률
            max_occurrences=params["max_occurrences"],              # k-mer 최대 발생 횟수
            covered_mask=mask,                # 기존 탐지 영역 마스크 (시딩 위치 스킵용)
            show_progress=False,              # 진행 상황 출력 비활성화
            label=f"{chromosome} Tier3",      # 진행 메시지용 레이블
        )  # 반환값: SeedCandidate 리스트 (미가공 반복 후보군)

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
