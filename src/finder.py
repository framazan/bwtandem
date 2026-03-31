import time  # 실행 시간 측정을 위한 표준 라이브러리
import numpy as np  # 수치 연산 및 배열 처리를 위한 NumPy
from typing import List, Tuple, Set, Optional, Dict  # 타입 힌트를 위한 typing 모듈
from .bwt_core import BWTCore  # BWT/FM-인덱스 핵심 구조체
from .models import TandemRepeat  # 반복 서열 데이터 클래스
from .tier1 import Tier1STRFinder  # Tier 1: 짧은 완전 반복 탐색기
from .tier2 import Tier2LCPFinder  # Tier 2: 중간 길이 불완전 반복 탐색기
from .tier3 import Tier3LongReadFinder  # Tier 3: 긴 반복 서열 탐색기
from .motif_utils import MotifUtils  # 모티프 정규화 및 통계 유틸리티

class TandemRepeatFinder:
    """Main coordinator for multi-tier tandem repeat finding."""

    VALID_TIERS = {"tier1", "tier2", "tier3"}  # 유효한 탐색 티어 이름 집합

    def __init__(self, sequence: str, chromosome: str = "chr1",
                 min_period: int = 1, max_period: int = 2000,
                 show_progress: bool = False,
                 enabled_tiers: Optional[Set[str]] = None,
                 min_array_bp: Optional[int] = None,
                 max_array_bp: Optional[int] = None,
                 tier3_mode: str = "balanced"):
        self.sequence = sequence  # 분석할 DNA 서열 문자열
        self.chromosome = chromosome  # 염색체 이름 (출력 레코드에 사용)
        self.min_period = min_period  # 탐색할 모티프의 최소 길이(bp)
        self.max_period = max_period  # 탐색할 모티프의 최대 길이(bp)
        self.show_progress = show_progress  # 진행 상황 출력 여부 플래그
        self.enabled_tiers = self._normalize_tiers(enabled_tiers)  # 활성화할 티어 집합 정규화

        # 배열 길이 하한/상한을 음수가 되지 않도록 처리
        self.min_array_bp = max(0, min_array_bp) if min_array_bp else None
        self.max_array_bp = max(0, max_array_bp) if max_array_bp else None
        if self.min_array_bp and self.max_array_bp and self.min_array_bp > self.max_array_bp:
            # Swap to keep bounds consistent
            # 하한이 상한보다 크면 서로 교환하여 일관성 유지
            self.min_array_bp, self.max_array_bp = self.max_array_bp, self.min_array_bp

        # Initialize BWT Core
        if show_progress:
            # BWT 인덱스 구축 시작을 사용자에게 알림
            print(f"  [{chromosome}] Building BWT and FM-index...", flush=True)
        t0 = time.time()  # BWT 구축 시작 시각 기록
        # Ensure sentinel
        if not sequence.endswith('$'):
            sequence += '$'  # BWT 구축을 위해 서열 끝에 센티넬 문자 '$' 추가

        self.bwt = BWTCore(sequence, sa_sample_rate=1)  # FM-인덱스(suffix array, BWT, occurrence 배열) 구축
        if show_progress:
            # BWT 구축 완료 및 소요 시간 출력
            print(f"  [{chromosome}] BWT built in {time.time() - t0:.2f}s", flush=True)

        # Initialize Tiers
        self.tier1 = None  # Tier 1 탐색기 초기화 (기본값 None)
        tier1_min = max(1, min_period)  # Tier 1이 다룰 모티프 최소 길이 (최소 1bp)
        tier1_max = min(9, max_period)  # Tier 1이 다룰 모티프 최대 길이 (최대 9bp)
        if "tier1" in self.enabled_tiers and tier1_min <= tier1_max:
            # Tier 1이 활성화되어 있고 유효한 범위가 있을 때만 인스턴스 생성
            self.tier1 = Tier1STRFinder(
                self.bwt.text_arr,          # 바이트 배열로 변환된 서열
                self.bwt,                   # FM-인덱스 객체
                max_motif_length=tier1_max, # 모티프 최대 길이
                min_motif_length=tier1_min, # 모티프 최소 길이
                allowed_mismatch_rate=0.2,  # 허용 미스매치 비율 20%
                allowed_indel_rate=0.1,     # 허용 삽입/결실 비율 10%
                show_progress=show_progress # 진행 상황 출력 여부 전달
            )

        self.tier2 = None  # Tier 2 탐색기 초기화 (기본값 None)
        if "tier2" in self.enabled_tiers:
            # Tier 2가 활성화되어 있을 때만 인스턴스 생성
            self.tier2 = Tier2LCPFinder(
                self.bwt,                   # FM-인덱스 객체
                min_period=min_period,      # 최소 모티프 길이
                max_period=max_period,      # 최대 모티프 길이
                allowed_mismatch_rate=0.2,  # 허용 미스매치 비율 20%
                allowed_indel_rate=0.1,     # 허용 삽입/결실 비율 10%
                show_progress=show_progress # 진행 상황 출력 여부 전달
            )

        # Tier 3이 활성화된 경우에만 인스턴스 생성, 그렇지 않으면 None
        self.tier3 = Tier3LongReadFinder(self.bwt, mode=tier3_mode) if "tier3" in self.enabled_tiers else None

    def find_all(self) -> List[TandemRepeat]:
        """Execute the full 3-tier finding pipeline."""
        all_repeats = []  # 모든 티어에서 발견된 반복 서열을 모을 리스트
        tier1_seen: Set[Tuple[int, int]] = set()  # Tier 1에서 발견된 (start, end) 좌표 집합 (중복 방지용)
        tier2_seen: Set[Tuple[int, int]] = set()  # Tier 2에서 발견된 (start, end) 좌표 집합 (중복 방지용)

        # --- Tier 1: Short Perfect Repeats ---
        if self.tier1:
            if self.show_progress:
                # Tier 1 시작 알림 출력
                print(f"  [{self.chromosome}] Running Tier 1 (Short Perfect)...", flush=True)
            t0 = time.time()  # Tier 1 시작 시각 기록
            tier1_repeats = self.tier1.find_strs(self.chromosome)  # Tier 1 실행: 짧은 완전 반복 탐색

            accepted = 0  # 필터를 통과한 반복 서열 개수 카운터
            for r in tier1_repeats:
                # 각 반복 서열을 전역 목록에 등록 (길이 범위 필터 포함)
                if self._register_repeat(r, all_repeats, tier1_seen):
                    accepted += 1  # 등록 성공 시 카운터 증가

            if self.show_progress:
                # Tier 1 완료 및 결과 요약 출력
                print(f"  [{self.chromosome}] Tier 1 found {accepted} repeats in {time.time() - t0:.2f}s", flush=True)
        elif self.show_progress:
            # Tier 1이 비활성화되었거나 범위를 벗어난 경우 알림
            print(f"  [{self.chromosome}] Skipping Tier 1 (disabled or out of requested range)", flush=True)

        # --- Tier 2: Imperfect & Medium Repeats (>=10bp by design) ---
        if self.tier2:
            if self.show_progress:
                # Tier 2 시작 알림 출력
                print(f"  [{self.chromosome}] Running Tier 2 (Imperfect & Medium)...", flush=True)
            t0 = time.time()  # Tier 2 시작 시각 기록
            # 2a. Long unit repeats (strict adjacency, >=20bp units)
            long_unit_repeats = []  # 긴 단위 반복 결과 리스트
            long_kept = 0  # 등록된 긴 단위 반복 개수
            min_unit = max(20, self.min_period)  # Tier 2 긴 단위 반복의 최소 모티프 길이 (최소 20bp)
            if self.max_period >= min_unit:
                # 최대 주기가 최소 단위 이상일 때만 긴 단위 반복 탐색 실행
                long_unit_repeats = self.tier2.find_long_unit_repeats_strict(
                    self.chromosome,          # 염색체 이름
                    min_unit_len=min_unit,    # 최소 단위 길이
                    max_unit_len=self.max_period,  # 최대 단위 길이
                    min_copies=2              # 최소 복제 수 2회
                )

                for r in long_unit_repeats:
                    is_new = True  # 새로운 반복 서열인지 여부 초기화
                    for start, end in tier1_seen:
                        # Tier 1에서 이미 발견된 위치와 겹치는지 확인
                        if not (r.end <= start or r.start >= end):
                            is_new = False  # 겹치는 경우 새로운 서열이 아님
                            break
                    if is_new and self._register_repeat(r, all_repeats, tier2_seen):
                        long_kept += 1  # 새로운 서열이고 등록 성공 시 카운터 증가

            # 2b. General scanning for medium/long repeats up to requested max
            medium_repeats = []  # 중간 길이 반복 결과 리스트
            medium_kept = 0  # 등록된 중간 길이 반복 개수
            # Force Tier2 to ignore classic microsatellites: start from period 10bp
            scan_lower = max(10, self.min_period)   # 스캔 하한: 마이크로새틀라이트 제외를 위해 최소 10bp
            scan_upper = min(50, self.max_period)   # 스캔 상한: 최대 50bp (BWT 시드 방식 적용 범위)
            if scan_upper >= scan_lower:
                # 유효한 스캔 범위가 있을 때만 실행
                combined_seen = tier1_seen.union(tier2_seen)  # Tier 1과 Tier 2에서 이미 발견된 위치 합집합
                medium_repeats = self.tier2.find_long_repeats(
                    self.chromosome,          # 염색체 이름
                    combined_seen,            # 중복 방지를 위한 이미 발견된 위치 집합
                    max_scan_period=scan_upper  # 스캔 최대 주기
                )

                for r in medium_repeats:
                    # 중간 길이 반복 서열 등록
                    if self._register_repeat(r, all_repeats, tier2_seen):
                        medium_kept += 1  # 등록 성공 시 카운터 증가
            if self.show_progress:
                total_kept = long_kept + medium_kept  # 긴 단위와 중간 길이 반복 합계
                # Tier 2 완료 및 결과 요약 출력
                print(f"  [{self.chromosome}] Tier 2 processed {total_kept} repeats (>=10bp motifs) in {time.time() - t0:.2f}s", flush=True)
        elif self.show_progress:
            # Tier 2가 비활성화된 경우 알림
            print(f"  [{self.chromosome}] Skipping Tier 2 (disabled)", flush=True)

        # --- Tier 3: Long Reads (Optional/Advanced) ---
        if self.tier3:
            if self.show_progress:
                # Tier 3 시작 알림 출력
                print(f"  [{self.chromosome}] Running Tier 3 (Long Reads)...", flush=True)
            t0 = time.time()  # Tier 3 시작 시각 기록

            combined_seen = tier1_seen.union(tier2_seen)  # Tier 1 + Tier 2 발견 위치 합집합 (Tier 3 중복 방지용)
            tier3_repeats = self.tier3.find_long_repeats(
                self.chromosome,  # 염색체 이름
                tier1_seen,       # Tier 1 발견 위치 집합
                tier2_seen        # Tier 2 발견 위치 집합
            )

            accepted = 0  # Tier 3에서 등록된 반복 서열 개수 카운터
            for r in tier3_repeats:
                # Tier 3 결과를 전역 목록에 등록
                if self._register_repeat(r, all_repeats, combined_seen):
                    accepted += 1  # 등록 성공 시 카운터 증가

            if self.show_progress:
                # Tier 3 완료 및 결과 요약 출력
                print(f"  [{self.chromosome}] Tier 3 found {accepted} repeats in {time.time() - t0:.2f}s", flush=True)
        elif self.show_progress:
            # Tier 3가 비활성화된 경우 알림
            print(f"  [{self.chromosome}] Skipping Tier 3 (disabled)", flush=True)

        # --- Post-processing ---
        # Sort by position
        all_repeats.sort(key=lambda x: x.start)  # 모든 반복 서열을 시작 위치 기준으로 정렬

        # Merge adjacent repeats (unify fragmented motifs)
        all_repeats = self._merge_adjacent_repeats(all_repeats)  # 같은 모티프를 가진 인접 반복 서열 병합

        # Filter overlaps (keep longest/best)
        final_repeats = self._filter_overlaps(all_repeats)  # 겹치는 반복 서열 중 점수가 높은 것만 유지
        final_repeats = [r for r in final_repeats if self._repeat_within_bounds(r)]  # 사용자 지정 범위 내 결과만 필터링

        return final_repeats  # 최종 반복 서열 목록 반환

    def _filter_overlaps(self, repeats: List[TandemRepeat]) -> List[TandemRepeat]:
        """Filter overlapping repeats, keeping the one with higher score/length."""
        if not repeats:
            return []  # 입력 리스트가 비어있으면 빈 리스트 반환

        # Sort by start position
        repeats.sort(key=lambda x: x.start)  # 시작 위치 기준으로 정렬

        filtered = [repeats[0]]  # 첫 번째 반복 서열을 결과 리스트에 추가

        for current in repeats[1:]:
            prev = filtered[-1]  # 현재까지 유지된 마지막 반복 서열

            # Check overlap
            if current.start < prev.end:
                # 두 반복 서열이 겹치는 경우 겹침 양 계산
                # Calculate overlap amount
                overlap = min(prev.end, current.end) - max(prev.start, current.start)  # 실제 겹치는 bp 수
                overlap_ratio = overlap / min(prev.length, current.length)  # 짧은 쪽 대비 겹침 비율

                if overlap_ratio > 0.5:  # Significant overlap
                    # 겹침이 50% 초과이면 점수가 더 높은 쪽을 선택
                    # Keep the better one
                    # Criteria: Length * (1 - mismatch_rate)
                    prev_score = prev.length * (1.0 - prev.mismatch_rate)   # 이전 반복 서열 점수 계산
                    curr_score = current.length * (1.0 - current.mismatch_rate)  # 현재 반복 서열 점수 계산

                    if curr_score > prev_score:
                        filtered[-1] = current  # 현재가 더 좋으면 이전 항목을 현재로 교체
                else:
                    # Small overlap, keep both (maybe compound?)
                    # 겹침이 적으면 두 반복 서열 모두 유지 (복합 반복일 수 있음)
                    filtered.append(current)
            else:
                filtered.append(current)  # 겹치지 않으면 그대로 추가

        return filtered  # 필터링된 반복 서열 목록 반환

    def _merge_adjacent_repeats(self, repeats: List[TandemRepeat]) -> List[TandemRepeat]:
        """Merge adjacent repeats with the same motif."""
        if not repeats:
            return []  # 입력 리스트가 비어있으면 빈 리스트 반환

        # Sort by start position
        repeats.sort(key=lambda x: x.start)  # 시작 위치 기준으로 정렬

        merged = [repeats[0]]  # 첫 번째 반복 서열로 병합 결과 리스트 시작

        for current in repeats[1:]:
            prev = merged[-1]  # 현재까지 병합된 마지막 항목

            # Check if they are adjacent or overlapping
            # Allow a small gap (e.g., up to period length or 10bp)
            gap = max(0, current.start - prev.end)  # 두 반복 서열 사이의 간격 계산 (음수 방지)

            # Check if motifs are compatible (same canonical motif)
            motif1 = prev.consensus_motif or prev.motif    # 이전 항목의 컨센서스 모티프 또는 원래 모티프
            motif2 = current.consensus_motif or current.motif  # 현재 항목의 컨센서스 모티프 또는 원래 모티프

            canon1, strand1 = MotifUtils.get_canonical_motif_stranded(motif1)  # 이전 모티프의 정규 형태와 가닥 방향
            canon2, strand2 = MotifUtils.get_canonical_motif_stranded(motif2)  # 현재 모티프의 정규 형태와 가닥 방향

            # Allow merge if canonical motifs match and gap is small
            max_gap = max(10, len(canon1))
            if canon1 == canon2 and gap <= max_gap:
                # Trial merge: check if combined region quality is acceptable
                new_start = min(prev.start, current.start)
                new_end = max(prev.end, current.end)
                avg_mm = (prev.mismatch_rate + current.mismatch_rate) / 2

                # Quick quality check: scan the merged region
                # Use actual motif from prediction (not canonical) for comparison
                text_arr = self.bwt.text_arr
                actual_motif = motif1
                motif_arr = np.frombuffer(actual_motif.encode('ascii'), dtype=np.uint8)
                mlen = len(actual_motif)
                trial_mismatches = 0
                trial_total = 0
                for pos in range(new_start, min(new_end, len(text_arr) - mlen), mlen):
                    window = text_arr[pos:pos + mlen]
                    if len(window) == mlen:
                        trial_mismatches += np.sum(window != motif_arr)
                        trial_total += mlen
                trial_mm = trial_mismatches / trial_total if trial_total > 0 else 0

                # Only merge if trial mismatch rate is reasonable
                # (not more than 2x the average of individual rates + 5% margin)
                max_acceptable_mm = max(avg_mm * 2, 0.15)
                if trial_mm <= max_acceptable_mm:
                    prev.start = new_start
                    prev.end = new_end
                    prev.length = new_end - new_start
                    prev.copies = prev.length / len(canon1)
                    self._recompute_stats(prev)
                else:
                    # Mismatch too high after merge — keep as separate repeats
                    merged.append(current)

            else:
                merged.append(current)  # 병합 조건을 만족하지 않으면 별도 항목으로 추가

        return merged  # 병합된 반복 서열 목록 반환

    def _recompute_stats(self, repeat: TandemRepeat):
        """Recompute statistics for a repeat (e.g. after merging)."""
        text_arr = self.bwt.text_arr  # FM-인덱스에서 바이트 배열 서열 참조
        motif = repeat.consensus_motif or repeat.motif  # 컨센서스 모티프 또는 원래 모티프 사용
        motif_len = len(motif)  # 모티프 길이 계산

        if motif_len == 0:
            return  # 모티프 길이가 0이면 통계 계산 불가, 즉시 반환

        # Re-derive consensus and mismatch rate from the full merged region
        # 병합된 전체 영역에서 컨센서스와 미스매치 비율 재계산
        consensus_arr, mm_rate, max_mm = MotifUtils.build_consensus_motif_array(
            text_arr, repeat.start, motif_len, int(repeat.copies)
        )

        if consensus_arr.size > 0:
            # 컨센서스 배열을 ASCII 문자열로 디코딩
            consensus_str = consensus_arr.tobytes().decode('ascii', errors='replace')
            repeat.consensus_motif = consensus_str  # 컨센서스 모티프 갱신
            repeat.motif = consensus_str # Update motif to new consensus  # 원래 모티프도 새 컨센서스로 갱신
            repeat.mismatch_rate = mm_rate  # 미스매치 비율 갱신
            repeat.max_mismatches_per_copy = max_mm  # 복제당 최대 미스매치 수 갱신

            # Recalculate TRF stats
            # TRF 호환 통계 재계산
            (percent_matches, percent_indels, score, composition,
             entropy, actual_sequence) = MotifUtils.calculate_trf_statistics(
                text_arr, repeat.start, repeat.end, consensus_str, int(repeat.copies), mm_rate
            )

            repeat.percent_matches = percent_matches  # 일치 비율 갱신
            repeat.percent_indels = percent_indels    # 삽입/결실 비율 갱신
            repeat.score = score                      # TRF 점수 갱신
            repeat.composition = composition          # 염기 조성 갱신
            repeat.entropy = entropy                  # 섀넌 엔트로피 갱신
            repeat.actual_sequence = actual_sequence  # 실제 서열 문자열 갱신

            # Update variations
            # 각 복제본의 변이 요약 갱신
            variations = MotifUtils.summarize_variations_array(
                text_arr, repeat.start, repeat.end, motif_len, consensus_arr
            )
            repeat.variations = variations  # 변이 정보 갱신

    def cleanup(self):
        """Release resources."""
        self.bwt.clear()  # BWT/FM-인덱스가 사용하는 메모리 해제

    def _normalize_tiers(self, tiers: Optional[Set[str]]) -> Set[str]:
        if not tiers:
            return set(self.VALID_TIERS)  # 티어가 지정되지 않으면 모든 유효 티어 활성화

        normalized: Set[str] = set()  # 정규화된 티어 이름 집합
        for tier in tiers:
            if not tier:
                continue  # 빈 문자열은 건너뜀
            name = tier.strip().lower()  # 공백 제거 및 소문자 변환으로 정규화
            if name == "all":
                return set(self.VALID_TIERS)  # "all"이면 모든 유효 티어 반환
            if name in self.VALID_TIERS:
                normalized.add(name)  # 유효한 티어 이름이면 집합에 추가

        return normalized if normalized else set(self.VALID_TIERS)  # 유효한 티어가 없으면 전체 티어 반환

    def _register_repeat(self, repeat: TandemRepeat, store: List[TandemRepeat],
                         seen: Optional[Set[Tuple[int, int]]] = None) -> bool:
        if not self._repeat_within_bounds(repeat):
            return False  # 사용자 지정 범위를 벗어난 반복 서열은 등록 거부

        store.append(repeat)  # 반복 서열을 전역 목록에 추가
        if seen is not None:
            seen.add((repeat.start, repeat.end))  # (start, end) 좌표를 중복 방지 집합에 등록
        return True  # 등록 성공 반환

    def _repeat_within_bounds(self, repeat: TandemRepeat) -> bool:
        motif = repeat.motif or repeat.consensus_motif  # 모티프 또는 컨센서스 모티프 사용
        motif_len = len(motif) if motif else 0  # 모티프 길이 계산 (없으면 0)
        if motif_len <= 0:
            motif_len = max(1, repeat.length)  # 모티프 길이가 0이면 배열 길이를 대신 사용

        if motif_len < self.min_period or motif_len > self.max_period:
            return False  # 모티프 길이가 허용 범위를 벗어나면 False

        length = repeat.length if repeat.length else repeat.end - repeat.start  # 배열 총 길이 계산

        if self.min_array_bp and length < self.min_array_bp:
            return False  # 배열 길이가 최소 bp 미만이면 False
        if self.max_array_bp and length > self.max_array_bp:
            return False  # 배열 길이가 최대 bp 초과이면 False

        return True  # 모든 조건을 통과하면 True 반환
