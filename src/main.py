import argparse  # 명령줄 인수 파싱을 위한 표준 라이브러리
import sys  # 시스템 종료(sys.exit) 및 표준 출력을 위한 라이브러리
import time  # 실행 시간 측정을 위한 표준 라이브러리
import os  # 파일 경로 처리 및 존재 여부 확인을 위한 라이브러리
from typing import List, Iterator, Tuple  # 타입 힌트를 위한 typing 모듈
from .finder import TandemRepeatFinder  # 멀티 티어 반복 서열 탐색 조율자
from .models import TandemRepeat  # 반복 서열 데이터 클래스 및 출력 포맷터


def _resolve_output_file(output_prefix: str, extension: str) -> str:
    """Return output path while avoiding duplicated extensions."""
    ext = extension if extension.startswith(".") else f".{extension}"  # 확장자 앞에 점이 없으면 추가
    if output_prefix.lower().endswith(ext.lower()):
        return output_prefix  # 이미 확장자가 포함된 경우 그대로 반환
    return f"{output_prefix}{ext}"  # 확장자를 접두사에 붙여 최종 파일 경로 반환

def parse_fasta(file_path: str) -> Iterator[Tuple[str, str]]:
    """Simple FASTA parser to avoid Biopython dependency."""
    name = None  # 현재 파싱 중인 시퀀스 이름 초기화
    seq_parts = []  # 현재 시퀀스의 라인별 조각을 모을 리스트
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()  # 앞뒤 공백 및 개행 문자 제거
            if not line:
                continue  # 빈 줄은 건너뜀
            if line.startswith('>'):
                # '>' 로 시작하는 헤더 라인 처리
                if name:
                    yield name, "".join(seq_parts)  # 이전 시퀀스를 완성하여 반환
                name = line[1:].split()[0]  # Take first word as ID  # '>'를 제거하고 첫 단어만 ID로 사용
                seq_parts = []  # 새 시퀀스의 조각 리스트 초기화
            else:
                seq_parts.append(line)  # 서열 라인을 조각 리스트에 추가
        if name:
            yield name, "".join(seq_parts)  # 파일 마지막 시퀀스 반환

def main():
    parser = argparse.ArgumentParser(description="BWT-based Tandem Repeat Finder")  # CLI 파서 생성
    parser.add_argument("fasta_file", help="Input FASTA file")  # 입력 FASTA 파일 경로 인수
    parser.add_argument("--min-period", type=int, default=1, help="Minimum period size (default: 1)")  # 최소 주기 길이 옵션
    parser.add_argument("--max-period", type=int, default=2000, help="Maximum period size (default: 2000)")  # 최대 주기 길이 옵션
    parser.add_argument("--min-array-bp", type=int, default=None,
                        help="Minimum repeat array length in bp (default: no minimum)")  # 반복 배열 최소 길이 옵션
    parser.add_argument("--max-array-bp", type=int, default=None,
                        help="Maximum repeat array length in bp (default: no maximum)")  # 반복 배열 최대 길이 옵션
    parser.add_argument("--tiers", type=str, default="tier1,tier2,tier3",
                        help="Comma-separated list of tiers to run (tier1,tier2,tier3) or 'all'")  # 실행할 티어 목록 옵션
    parser.add_argument("--output", "-o", help="Output file prefix (default: input filename)")  # 출력 파일 접두사 옵션
    parser.add_argument("--format", choices=["bed", "vcf", "trf", "strfinder"], default="bed", help="Output format")  # 출력 형식 선택 옵션
    parser.add_argument("--verbose", "-v", action="store_true", help="Show progress")  # 상세 출력 모드 플래그
    parser.add_argument("--profile", action="store_true", help="Profile execution with cProfile and print top hotspots")  # 성능 프로파일링 플래그
    parser.add_argument("--tier3-mode", choices=["fast", "balanced", "sensitive"],
                        default="balanced", help="Tier 3 speed/accuracy preset (default: balanced)")  # Tier 3 속도/정확도 사전 설정 옵션

    args = parser.parse_args()  # 명령줄 인수 파싱 실행

    if not os.path.exists(args.fasta_file):
        # 입력 파일이 존재하지 않으면 오류 메시지 출력 후 종료
        print(f"Error: File {args.fasta_file} not found")
        sys.exit(1)

    output_prefix = args.output if args.output else os.path.splitext(args.fasta_file)[0]  # 출력 접두사: 지정 없으면 입력 파일명 (확장자 제외)
    out_file = _resolve_output_file(output_prefix, args.format)  # 최종 출력 파일 경로 결정

    print(f"Processing {args.fasta_file}...")  # 처리 시작 알림
    start_total = time.time()  # 전체 처리 시작 시각 기록

    all_repeats: List[TandemRepeat] = []  # 모든 염색체의 반복 서열 결과를 모을 리스트

    tiers_arg = args.tiers.strip()  # 티어 인수 앞뒤 공백 제거
    if tiers_arg.lower() == "all":
        enabled_tiers = {"tier1", "tier2", "tier3"}  # "all"이면 모든 티어 활성화
    else:
        # 콤마로 구분된 티어 이름을 소문자로 정규화하여 집합 생성
        enabled_tiers = {t.strip().lower() for t in tiers_arg.split(',') if t.strip()}

    # Optional profiler
    profiler = None  # 프로파일러 초기화 (기본값 None)
    if args.profile:
        import cProfile  # 성능 프로파일링을 위한 cProfile 동적 임포트
        profiler = cProfile.Profile()  # 프로파일러 인스턴스 생성
        profiler.enable()  # 프로파일링 시작

    for chrom, seq in parse_fasta(args.fasta_file):
        seq = seq.upper()  # DNA 서열을 대문자로 변환 (일관성 보장)

        if args.verbose:
            # 상세 모드: 현재 처리 중인 시퀀스 이름과 길이 출력
            print(f"Processing sequence: {chrom} ({len(seq)} bp)")

        finder = TandemRepeatFinder(
            seq,                          # 분석할 서열
            chromosome=chrom,             # 염색체 이름
            min_period=args.min_period,   # 최소 모티프 길이
            max_period=args.max_period,   # 최대 모티프 길이
            show_progress=args.verbose,   # 진행 상황 출력 여부
            enabled_tiers=enabled_tiers,  # 활성화할 티어 집합
            min_array_bp=args.min_array_bp,  # 반복 배열 최소 길이 필터
            max_array_bp=args.max_array_bp,  # 반복 배열 최대 길이 필터
            tier3_mode=args.tier3_mode,   # Tier 3 속도/정확도 모드
        )

        repeats = finder.find_all()  # 멀티 티어 파이프라인 실행하여 반복 서열 탐색
        all_repeats.extend(repeats)  # 현재 염색체 결과를 전체 결과 목록에 추가

        finder.cleanup()  # BWT/FM-인덱스 메모리 해제

    # Stop profiler and report
    if profiler is not None:
        profiler.disable()  # 프로파일링 종료

    print(f"Total repeats found: {len(all_repeats)}")  # 발견된 총 반복 서열 수 출력
    print(f"Total time: {time.time() - start_total:.2f}s")  # 전체 처리 시간 출력

    if profiler is not None:
        import pstats  # 프로파일 통계 출력을 위한 pstats 동적 임포트
        profile_path = f"{output_prefix}.tier2_profile.prof"  # 프로파일 결과 파일 경로
        profiler.dump_stats(profile_path)  # 프로파일 데이터를 파일에 저장
        print(f"Profile written to {profile_path}")  # 저장 경로 출력
        print("Top 20 cumulative time hotspots:")  # 누적 시간 기준 상위 20개 핫스팟 안내
        stats = pstats.Stats(profiler)  # 프로파일 통계 객체 생성
        stats.strip_dirs().sort_stats("cumulative").print_stats(20)  # 디렉터리 제거 후 누적 시간 정렬하여 상위 20개 출력

    # Write output
    if args.format == "bed":
        out_file = _resolve_output_file(output_prefix, "bed")  # BED 형식 출력 파일 경로 결정
        with open(out_file, "w") as f:
            for r in all_repeats:
                f.write(r.to_bed() + "\n")  # 각 반복 서열을 BED 형식으로 파일에 기록
    elif args.format == "vcf":
        out_file = _resolve_output_file(output_prefix, "vcf")  # VCF 형식 출력 파일 경로 결정
        with open(out_file, "w") as f:
            f.write("##fileformat=VCFv4.2\n")  # VCF 파일 형식 헤더 기록
            f.write("##INFO=<ID=END,Number=1,Type=Integer,Description=\"End position of the repeat\">\n")  # INFO 필드 정의 헤더
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")  # VCF 열 헤더 기록
            for r in all_repeats:
                # Use a single anchor base for REF to keep symbolic-ALT records valid.
                # 심볼릭 ALT 레코드 유효성을 위해 REF에 단일 앵커 염기 사용
                ref = "N"  # 기본 REF 값 (서열 정보 없을 경우)
                if r.actual_sequence:
                    ref = r.actual_sequence[0]  # 실제 서열의 첫 번째 염기를 REF로 사용
                elif r.consensus_motif:
                    ref = r.consensus_motif[0]  # 컨센서스 모티프의 첫 염기를 REF로 사용
                elif r.motif:
                    ref = r.motif[0]  # 모티프의 첫 염기를 REF로 사용
                alt = "<STR>"  # 반복 서열을 나타내는 심볼릭 ALT 값
                info = f"END={r.end};{r.to_vcf_info()}"  # END 위치와 VCF INFO 필드 조합
                f.write(f"{r.chrom}\t{r.start+1}\t.\t{ref}\t{alt}\t.\t.\t{info}\n")  # 1-based 좌표로 VCF 레코드 기록
    elif args.format == "trf":
        out_file = _resolve_output_file(output_prefix, "dat")  # TRF .dat 형식 출력 파일 경로 결정
        with open(out_file, "w") as f:
            for r in all_repeats:
                f.write(r.to_trf_dat() + "\n")  # 각 반복 서열을 TRF .dat 형식으로 파일에 기록
    elif args.format == "strfinder":
        out_file = _resolve_output_file(output_prefix, "csv")  # STRfinder CSV 형식 출력 파일 경로 결정
        with open(out_file, "w") as f:
            # STRfinder CSV 헤더 기록
            f.write("STR_marker,STR_position,STR_motif,STR_genotype_structure,STR_genotype,STR_core_seq,Allele_coverage,Alleles_ratio,Reads_Distribution,STR_depth,Full_seq,Variations\n")
            for r in all_repeats:
                f.write(r.to_strfinder() + "\n")  # 각 반복 서열을 STRfinder 형식으로 파일에 기록

    print(f"Results written to {out_file}")  # 결과 파일 저장 경로 출력

if __name__ == "__main__":
    main()  # 스크립트 직접 실행 시 main 함수 호출
