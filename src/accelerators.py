"""Optional accelerators backed by the Cython extension."""
from __future__ import annotations  # 파이썬 3.9 이전 버전에서도 타입 힌트 전방 참조 허용

from typing import Optional, Tuple  # 타입 힌트를 위한 Optional, Tuple 임포트

import numpy as np  # 수치 배열 처리를 위한 NumPy 임포트

try:
    from . import _accelerators as _native  # type: ignore  # 컴파일된 Cython 확장 모듈 임포트 시도
except Exception:  # pragma: no cover - fallback to pyximport
    # Cython .so 파일이 없을 경우 pyximport를 통해 런타임 컴파일 시도
    _native = None  # 네이티브 모듈 초기화 실패로 None 설정
    try:
        import pyximport  # type: ignore  # 런타임 Cython 컴파일러 임포트

        pyximport.install(  # type: ignore[attr-defined]
            language_level=3,           # Python 3 문법 사용
            inplace=True,               # 소스 디렉터리에 빌드
            build_in_temp=False,        # 임시 디렉터리 사용 안 함
            setup_args={"include_dirs": np.get_include()},  # NumPy 헤더 경로 포함
        )
        from . import _accelerators as _native  # type: ignore  # pylint: disable=import-error  # pyximport로 컴파일 후 재임포트
    except Exception as e:
        print(f"DEBUG: importing _accelerators failed: {e}")  # 임포트 실패 원인 디버그 출력
        _native = None  # 모든 시도 실패 시 None으로 설정 → 순수 파이썬 폴백 사용


AcceleratorResult = Optional[Tuple[int, int, int, int, int]]  # 가속기 함수 반환 타입 별칭 (start, end, copies 등)


if _native is not None:
    # Direct alias for maximum performance
    # Cython 네이티브 함수에 직접 별칭을 설정하여 최대 성능 확보
    hamming_distance = _native.hamming_distance        # 두 배열 간 해밍 거리 계산 함수
    extend_with_mismatches = _native.extend_with_mismatches  # 미스매치를 허용하며 반복 경계를 확장하는 함수
    pack_sequence = _native.pack_sequence              # DNA 서열을 2비트 팩 형식으로 압축하는 함수

    # Wrap scan_unit_repeats to ensure it appears in profiler output
    # (The overhead is negligible as it's called once per unit_len)
    # 프로파일러에서 식별 가능하도록 래퍼 함수로 감쌈 (unit_len당 1회 호출이므로 오버헤드 무시)
    def scan_unit_repeats(
        text_arr: np.ndarray,               # 분석할 서열 바이트 배열
        n: int,                             # 서열 유효 길이 (센티넬 제외)
        unit_len: int,                      # 탐색할 반복 단위(모티프) 길이
        min_copies: int,                    # 반복으로 인정할 최소 복제 수
        max_mismatch: int,                  # 허용할 최대 미스매치 수
        packed_arr: Optional[np.ndarray] = None  # 사전에 팩 처리된 서열 배열 (선택적)
    ) -> list:
        return _native.scan_unit_repeats(text_arr, n, unit_len, min_copies, max_mismatch, packed_arr)
        # Cython 구현의 scan_unit_repeats 호출하여 반복 서열 위치 목록 반환

    def scan_simple_repeats(
        text_arr: np.ndarray,               # 분석할 서열 바이트 배열
        tier1_mask: np.ndarray,             # Tier 1에서 이미 발견된 위치를 나타내는 마스크 배열
        n: int,                             # 서열 유효 길이
        min_p: int,                         # 탐색 최소 주기 길이
        max_p: int,                         # 탐색 최대 주기 길이
        period_step: int,                   # 주기 탐색 간격 (스텝 크기)
        position_step: int,                 # 위치 탐색 간격 (스텝 크기)
        allowed_mismatch_rate: float        # 허용 미스매치 비율
    ) -> list:
        return _native.scan_simple_repeats(
            text_arr, tier1_mask, n, min_p, max_p, period_step, position_step, allowed_mismatch_rate
        )  # Cython 구현의 슬라이딩 윈도우 반복 스캔 호출

    def find_periodic_patterns(
        positions: np.ndarray,              # FM-인덱스에서 찾은 k-mer 발생 위치 배열
        min_period: int,                    # 탐색 최소 주기
        max_period: int,                    # 탐색 최대 주기
        min_copies: int,                    # 최소 복제 수
        tolerance_ratio: float = 0.01       # 등차수열 허용 오차 비율
    ) -> list:
        return _native.find_periodic_patterns(positions, min_period, max_period, min_copies, tolerance_ratio)
        # Cython 구현: 위치 배열에서 등차수열(주기적 패턴)을 탐지하여 반환

    def find_periodic_runs(
        positions: np.ndarray,              # FM-인덱스에서 찾은 k-mer 발생 위치 배열
        min_period: int,                    # 탐색 최소 주기
        max_period: int,                    # 탐색 최대 주기
        min_copies: int,                    # 최소 복제 수
        tolerance_ratio: float = 0.01       # 허용 오차 비율
    ) -> list:
        return _native.find_periodic_runs(positions, min_period, max_period, min_copies, tolerance_ratio)
        # Cython 구현: 연속적인 주기 런(run)을 찾아 반환

    def align_unit_to_window(
        motif: bytes,                       # 정렬 기준 모티프 바이트 문자열
        window: bytes,                      # 비교할 서열 윈도우 바이트 문자열
        max_indel: int,                     # 허용할 최대 삽입/결실 수
        mismatch_tolerance: int             # 허용할 최대 미스매치 수
    ) -> Optional[Tuple]:
        return _native.align_unit_to_window(motif, window, max_indel, mismatch_tolerance)
        # Cython 구현: 모티프와 윈도우 간 정렬 수행 후 결과 반환

    def lcp_tandem_candidates(
        sa: np.ndarray,                     # 접미사 배열(suffix array)
        lcp: np.ndarray,                    # LCP(longest common prefix) 배열
        n: int,                             # 서열 유효 길이
        min_period: int,                    # 탐색 최소 주기
        max_period: int,                    # 탐색 최대 주기
        min_lcp_threshold: int = 10         # LCP 값의 최소 임계값 (노이즈 필터링)
    ) -> list:
        return _native.lcp_tandem_candidates(sa, lcp, n, min_period, max_period, min_lcp_threshold)
        # Cython 구현: LCP 배열에서 탠덤 반복 후보 (주기, 시작위치) 목록 반환

    def find_tandem_runs(
        positions: np.ndarray,              # k-mer 발생 위치 배열
        period: int,                        # 탐색할 주기 길이
        min_copies: int                     # 최소 복제 수
    ) -> list:
        return _native.find_tandem_runs(positions, period, min_copies)
        # Cython 구현: 지정 주기의 탠덤 런을 위치 배열에서 찾아 반환

    def anchor_scan_boundaries(
        text_arr: np.ndarray,               # 서열 바이트 배열
        seed_pos: int,                      # 시드(앵커) 위치
        period: int,                        # 반복 주기 길이
        n: int,                             # 서열 유효 길이
        match_threshold: float,             # 복제본 일치 비율 임계값
        max_backward_periods: int,          # 역방향 탐색 최대 주기 수
        max_forward_periods: int,           # 순방향 탐색 최대 주기 수
    ) -> Tuple[int, int]:
        return _native.anchor_scan_boundaries(
            text_arr, seed_pos, period, n, match_threshold,
            max_backward_periods, max_forward_periods
        )  # Cython 구현: 앵커에서 양방향으로 반복 배열 경계를 스캔하여 (시작, 끝) 반환
else:
    # Cython 모듈 없을 때 순수 파이썬 폴백 함수들 정의
    def hamming_distance(arr1: np.ndarray, arr2: np.ndarray) -> Optional[int]:
        return None  # 폴백: 해밍 거리 계산 불가, None 반환

    def extend_with_mismatches(
        s_arr: np.ndarray,                  # 서열 배열
        start_pos: int,                     # 탐색 시작 위치
        period: int,                        # 반복 주기
        n: int,                             # 서열 유효 길이
        allowed_mismatch_rate: float,       # 허용 미스매치 비율
    ) -> AcceleratorResult:
        return None  # 폴백: 미스매치 허용 확장 불가, None 반환

    def pack_sequence(text_arr: np.ndarray) -> np.ndarray:
        return np.array([], dtype=np.uint8)  # 폴백: 빈 배열 반환 (팩 처리 불가)

    def scan_unit_repeats(
        text_arr: np.ndarray,               # 서열 배열
        n: int,                             # 서열 유효 길이
        unit_len: int,                      # 반복 단위 길이
        min_copies: int,                    # 최소 복제 수
        max_mismatch: int,                  # 최대 허용 미스매치 수
        packed_arr: Optional[np.ndarray] = None  # 팩 처리된 배열 (선택적)
    ) -> list:
        return []  # 폴백: 빈 결과 반환 (Cython 없이 대규모 스캔 불가)

    def scan_simple_repeats(
        text_arr: np.ndarray,               # 서열 배열
        tier1_mask: np.ndarray,             # Tier 1 마스크 배열
        n: int,                             # 서열 유효 길이
        min_p: int,                         # 최소 주기
        max_p: int,                         # 최대 주기
        period_step: int,                   # 주기 스텝
        position_step: int,                 # 위치 스텝
        allowed_mismatch_rate: float        # 허용 미스매치 비율
    ) -> list:
        return []  # 폴백: 빈 결과 반환

    def find_periodic_patterns(
        positions: np.ndarray,              # 위치 배열
        min_period: int,                    # 최소 주기
        max_period: int,                    # 최대 주기
        min_copies: int,                    # 최소 복제 수
        tolerance_ratio: float = 0.01       # 허용 오차 비율
    ) -> list:
        return []  # 폴백: 빈 결과 반환

    def align_unit_to_window(
        motif: bytes,                       # 모티프 바이트 문자열
        window: bytes,                      # 윈도우 바이트 문자열
        max_indel: int,                     # 최대 삽입/결실 수
        mismatch_tolerance: int             # 최대 미스매치 수
    ) -> Optional[Tuple]:
        return None  # 폴백: 정렬 수행 불가, None 반환

    def find_periodic_runs(
        positions: np.ndarray,              # 위치 배열
        min_period: int,                    # 최소 주기
        max_period: int,                    # 최대 주기
        min_copies: int,                    # 최소 복제 수
        tolerance_ratio: float = 0.01       # 허용 오차 비율
    ) -> list:
        return []  # 폴백: 빈 결과 반환

    def lcp_tandem_candidates(
        sa: np.ndarray,                     # 접미사 배열
        lcp: np.ndarray,                    # LCP 배열
        n: int,                             # 서열 유효 길이
        min_period: int,                    # 최소 주기
        max_period: int,                    # 최대 주기
        min_lcp_threshold: int = 10         # LCP 최소 임계값
    ) -> list:
        """Pure-Python fallback for LCP tandem candidate detection."""
        results = []  # 탠덤 반복 후보 결과 리스트
        sa_len = len(sa)    # 접미사 배열 길이
        lcp_len = len(lcp)  # LCP 배열 길이
        limit = min(sa_len, lcp_len)  # 두 배열 중 짧은 쪽 길이로 순회 범위 제한
        for i in range(1, limit):
            L = int(lcp[i])  # 현재 위치의 LCP 값
            if L < min_lcp_threshold:
                continue  # LCP 값이 임계값보다 작으면 노이즈로 간주하고 건너뜀
            pos_a = int(sa[i - 1])  # 이전 접미사의 시작 위치
            pos_b = int(sa[i])      # 현재 접미사의 시작 위치
            if pos_a >= n or pos_b >= n:
                continue  # 센티넬 '$' 이후 위치이면 건너뜀
            diff = abs(pos_b - pos_a)  # 두 접미사 간 위치 차이 = 잠재적 주기 길이
            if diff < min_period or diff > max_period:
                continue  # 주기 범위를 벗어나면 건너뜀
            start = min(pos_a, pos_b)  # 두 위치 중 작은 값이 반복 시작 위치
            results.append((diff, start))  # (주기, 시작위치) 튜플을 결과에 추가
        return results  # 탠덤 반복 후보 목록 반환

    def find_tandem_runs(
        positions: np.ndarray,              # k-mer 발생 위치 배열
        period: int,                        # 탐색할 주기
        min_copies: int                     # 최소 복제 수
    ) -> list:
        """Pure-Python fallback for tandem run detection."""
        n_pos = len(positions)  # 위치 배열의 원소 수
        if n_pos < min_copies:
            return []  # 위치 수가 최소 복제 수보다 적으면 빈 결과 반환
        results = []  # 탠덤 런 결과 리스트
        run_start = int(positions[0])  # 현재 런의 시작 위치
        expected_next = run_start + period  # 다음 위치의 예상 값 (등차수열)
        count = 1  # 현재 런의 길이 카운터
        for i in range(1, n_pos):
            if int(positions[i]) == expected_next:
                # 예상 위치와 일치하면 런 계속
                count += 1  # 복제 수 증가
                expected_next = int(positions[i]) + period  # 다음 예상 위치 갱신
            else:
                # 연속성이 끊긴 경우 이전 런을 저장하고 새 런 시작
                if count >= min_copies:
                    results.append((run_start, expected_next))  # 충분한 복제 수이면 결과에 추가
                run_start = int(positions[i])  # 새 런의 시작 위치
                expected_next = run_start + period  # 새 런의 다음 예상 위치
                count = 1  # 카운터 초기화
        if count >= min_copies:
            results.append((run_start, expected_next))  # 마지막 런 처리
        return results  # 탠덤 런 목록 반환

    def anchor_scan_boundaries(
        text_arr: np.ndarray,               # 서열 바이트 배열
        seed_pos: int,                      # 시드(앵커) 위치
        period: int,                        # 반복 주기 길이
        n: int,                             # 서열 유효 길이
        match_threshold: float,             # 복제본 일치 비율 임계값
        max_backward_periods: int,          # 역방향 탐색 최대 주기 수
        max_forward_periods: int,           # 순방향 탐색 최대 주기 수
    ) -> Tuple[int, int]:
        """Pure-Python fallback for anchor-based boundary scanning."""
        if seed_pos + period > n:
            return (seed_pos, seed_pos + period)  # 시드가 서열 끝에 있으면 최소 범위 반환

        motif_arr = text_arr[seed_pos:seed_pos + period]  # 시드 위치의 모티프 배열 추출
        true_start = seed_pos        # 반복 배열의 실제 시작 위치 (초기값: 시드 위치)
        true_end = seed_pos + period  # 반복 배열의 실제 끝 위치 (초기값: 시드 + 주기)

        # Scan backward
        # 시드에서 역방향(5' 방향)으로 반복 경계 탐색
        scan_start = max(0, seed_pos - period * max_backward_periods)  # 역방향 탐색 한계 위치
        pos = seed_pos - period  # 역방향 탐색 시작 위치
        while pos >= scan_start:
            window = text_arr[pos:pos + period]  # 현재 위치의 윈도우 배열 추출
            if window.size == period:
                # 윈도우가 모티프와 일치하는 염기 수 계산
                matches = int(np.sum(window == motif_arr))
                if matches / period >= match_threshold:
                    # 일치 비율이 임계값 이상이면 시작 위치를 역방향으로 확장
                    true_start = pos
                    pos -= period  # 한 주기 더 역방향으로 이동
                else:
                    break  # 일치 비율 미달 시 역방향 탐색 종료
            else:
                break  # 윈도우 크기가 맞지 않으면 종료

        # Scan forward
        # 시드에서 순방향(3' 방향)으로 반복 경계 탐색
        scan_end = min(n, seed_pos + period * max_forward_periods)  # 순방향 탐색 한계 위치
        pos = seed_pos + period  # 순방향 탐색 시작 위치
        while pos + period <= scan_end:
            window = text_arr[pos:pos + period]  # 현재 위치의 윈도우 배열 추출
            if window.size == period:
                # 윈도우가 모티프와 일치하는 염기 수 계산
                matches = int(np.sum(window == motif_arr))
                if matches / period >= match_threshold:
                    # 일치 비율이 임계값 이상이면 끝 위치를 순방향으로 확장
                    true_end = pos + period
                    pos += period  # 한 주기 더 순방향으로 이동
                else:
                    break  # 일치 비율 미달 시 순방향 탐색 종료
            else:
                break  # 윈도우 크기가 맞지 않으면 종료

        return (true_start, true_end)  # 탐색된 반복 배열의 (시작, 끝) 좌표 반환
