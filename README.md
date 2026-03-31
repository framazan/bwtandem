# BWT 기반 직렬 반복 서열 탐지기 (BWT-based Tandem Repeat Finder)

## 프로젝트 개요

BWT Tandem Repeat Finder는 게놈 FASTA 파일에서 직렬 반복 서열(Tandem Repeat)을 탐지하는 도구입니다. Burrows-Wheeler Transform(BWT)과 FM-index를 핵심 알고리즘으로 사용하며, 짧은 완벽 반복(STR/마이크로새틀라이트)부터 수백 kb에 달하는 초장 반복까지 3단계 파이프라인으로 포괄적으로 탐지합니다.

### 왜 이 도구가 필요한가?

직렬 반복 서열은 유전체 불안정성, 유전 질환, 진화 연구에서 중요한 역할을 합니다. 기존 도구(TRF 등)는 긴 반복이나 불완전한 반복 탐지에 한계가 있습니다. 이 도구는 다음과 같은 특성을 갖습니다:

- **FM-index 기반 고속 탐색**: 접미사 배열과 BWT를 이용해 대용량 염색체 서열을 효율적으로 처리
- **3-Tier 파이프라인**: 반복 유형에 최적화된 3단계 탐지 구조로 탐지율과 속도를 동시에 달성
- **다양한 출력 형식**: BED, VCF, TRF `.dat`, STRfinder `.csv` 지원
- **적응형 파라미터**: 서열 길이·GC 함량·커버리지에 따라 Tier 3 파라미터를 자동 조정
- **선택적 가속화**: Cython 확장 및 Numba JIT 컴파일로 핵심 연산 가속화 가능

---

## 설치 방법

### Micromamba 환경 설정 (권장)

[Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)를 사용하면 모든 의존성을 격리된 환경에서 관리할 수 있습니다.

```bash
# 1. 환경 생성 (Python 3.11 + 주요 의존성)
micromamba create -n bwtandem python=3.11 numpy cython pytest numba setuptools -c conda-forge -y

# 2. pydivsufsort 설치 (conda-forge에 없으므로 pip으로 설치, --no-build-isolation 필요)
micromamba run -n bwtandem pip install pydivsufsort --no-build-isolation

# 3. Cython 확장 빌드 (GCC 4.x 환경에서는 -std=c99 필요)
micromamba run -n bwtandem python3 -c "
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
ext_modules = [Extension('src._accelerators', ['src/_accelerators.pyx'],
               include_dirs=[np.get_include()], extra_compile_args=['-std=c99'])]
setup(script_args=['build_ext', '--inplace'],
      ext_modules=cythonize(ext_modules, compiler_directives={'language_level': '3'}))
"

# 4. 테스트 실행
micromamba run -n bwtandem python3 -m pytest tests/ -v

# 5. 도구 실행
micromamba run -n bwtandem python3 -m src.main input.fa --format bed -v
```

### Python 의존성 직접 설치

Micromamba 없이 직접 설치할 수도 있습니다.

```bash
# 필수 의존성
pip install numpy pydivsufsort

# 선택 의존성 (성능 향상)
pip install numba Cython
```

| 패키지 | 구분 | 설명 |
|--------|------|------|
| `numpy` | 필수 | 배열 연산 |
| `pydivsufsort` | 필수 (권장) | 고속 접미사 배열 생성; 없으면 NumPy prefix-doubling 방식으로 대체 |
| `numba` | 선택 | rank 쿼리 및 LCP 계산 JIT 가속화 |
| `Cython` | 선택 | `_accelerators.pyx` 컴파일로 핵심 경로 가속화 |

### Cython 확장 빌드

`_accelerators.pyx`는 Hamming 거리 계산, 미스매치 연장, LCP 후보 탐지, DP 정렬 등 성능 핵심 경로를 제공합니다. 컴파일하지 않으면 순수 Python 폴백이 사용되며, 일부 코드 경로는 빈 결과를 반환합니다.

```bash
# 프로젝트 루트에서 실행
python3 -c "
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
ext_modules = [Extension('src._accelerators', ['src/_accelerators.pyx'], include_dirs=[np.get_include()])]
setup(script_args=['build_ext', '--inplace'], ext_modules=cythonize(ext_modules, compiler_directives={'language_level': '3'}))
"
```

빌드가 성공하면 `src/_accelerators.*.so` 파일이 생성됩니다.

### Singularity 컨테이너

모든 의존성과 Cython 확장이 포함된 컨테이너를 빌드할 수 있습니다.

```bash
# 컨테이너 빌드 (관리자 권한 필요)
sudo singularity build bwtandem.sif Singularity

# 컨테이너로 실행
singularity exec bwtandem.sif python3 -m src.main input.fa --format bed
```

컨테이너 이미지는 Ubuntu 22.04 기반이며 numpy, numba, pydivsufsort, Cython이 사전 설치되어 있습니다.

---

## 빠른 시작

```bash
# 가장 기본적인 실행 (기본값: 모든 Tier, BED 출력)
python3 -m src.main input.fa

# 결과 확인
cat input.bed
```

위 명령으로 `input.bed` 파일이 생성됩니다. BED 형식은 `chrom`, `start`, `end`, `motif`, `copies`, `tier`, `mismatch_rate`, `strand` 열을 포함합니다.

---

## CLI 옵션 상세

```
python3 -m src.main <fasta_file> [options]
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `fasta_file` | (필수) | 입력 FASTA 파일 경로 |
| `--min-period INT` | `1` | 탐지할 최소 반복 단위 길이 (bp) |
| `--max-period INT` | `2000` | 탐지할 최대 반복 단위 길이 (bp) |
| `--min-array-bp INT` | 없음 | 탐지 결과의 최소 반복 어레이 길이 필터 (bp) |
| `--max-array-bp INT` | 없음 | 탐지 결과의 최대 반복 어레이 길이 필터 (bp) |
| `--tiers TIERS` | `tier1,tier2,tier3` | 실행할 Tier 목록 (쉼표 구분, 또는 `all`) |
| `--tier3-mode MODE` | `balanced` | Tier 3 속도/정확도 프리셋: `fast`, `balanced`, `sensitive` |
| `--format FORMAT` | `bed` | 출력 형식: `bed`, `vcf`, `trf`, `strfinder` |
| `-o, --output PREFIX` | 입력 파일명 | 출력 파일 접두사 |
| `-t, --threads INT` | `1` | 병렬 처리 프로세스 수 (염색체별 병렬 처리) |
| `--mask MODE` | `none` | 마스킹 모드: `none`, `soft`, `hard`, `both` |
| `-v, --verbose` | `False` | 진행 상황 출력 |
| `--profile` | `False` | cProfile로 실행 시간 분석 후 상위 핫스팟 출력 |

### 옵션 상세 설명

**`--min-period` / `--max-period`**
반복 단위(모티프)의 길이 범위를 지정합니다. Tier 1은 1–9 bp, Tier 2는 10 bp 이상, Tier 3은 100 bp 이상을 각각 담당하므로, `--max-period 100`으로 설정하면 Tier 3의 효과가 제한됩니다.

**`--min-array-bp` / `--max-array-bp`**
반복 어레이 전체 길이(반복 단위 × 복사 수)를 기준으로 결과를 필터링합니다. 예: `--min-array-bp 30`은 30 bp 미만의 짧은 어레이를 제거합니다.

**`--tiers`**
특정 Tier만 실행할 때 사용합니다. `tier1`, `tier2`, `tier3` 또는 `all`을 조합할 수 있습니다.

**`--tier3-mode`**
- `fast`: 큰 k-mer 크기, 넓은 stride로 빠른 탐색 (민감도 감소)
- `balanced`: 기본값, 속도와 정확도의 균형
- `sensitive`: 작은 k-mer 크기, 좁은 stride로 세밀한 탐색 (속도 감소)

**`-t, --threads`**
멀티 프로세스로 여러 염색체를 동시에 처리합니다. 각 염색체가 독립적으로 BWT 인덱스를 구축하고 반복 서열을 탐지하므로, 다중 염색체 FASTA 파일에서 선형적 속도 향상을 기대할 수 있습니다. 단일 염색체 파일에서는 효과가 없습니다.

```bash
# 4개 프로세스로 병렬 처리
python3 -m src.main genome.fa -t 4 -v

# CPU 코어 수에 맞게 설정 (nproc 명령 활용)
python3 -m src.main genome.fa -t $(nproc) -v
```

**`--mask`**
게놈 FASTA 파일의 소프트 마스크(lowercase)와 하드 마스크(N) 영역을 처리하는 방법을 지정합니다.

| 모드 | 소문자 (acgt) | N 문자 | 설명 |
|------|-------------|-------|------|
| `none` | 대문자로 변환 | 유지 | 기본값: 마스크 무시, 모든 서열 분석 |
| `soft` | N으로 치환 | 유지 | 소프트 마스크 영역(반복 원소 등)을 건너뜀 |
| `hard` | 대문자로 변환 | 유지 | 하드 마스크 영역(갭, 미지정 영역)만 건너뜀 |
| `both` | N으로 치환 | 유지 | 소프트+하드 마스크 영역 모두 건너뜀 |

소프트 마스크는 RepeatMasker 등의 도구에서 반복 원소(LINE, SINE, transposon 등)를 소문자로 표기하는 관례를 따릅니다. 이미 알려진 산재 반복(interspersed repeat)을 제외하고 직렬 반복만 분석하고 싶을 때 `--mask soft`를 사용합니다.

```bash
# 소프트 마스크 영역 제외하고 분석
python3 -m src.main masked_genome.fa --mask soft -v

# 소프트+하드 마스크 모두 제외
python3 -m src.main masked_genome.fa --mask both -v
```

---

## 3-Tier 탐지 파이프라인

`TandemRepeatFinder` 코디네이터가 염색체별로 `BWTCore` FM-index를 한 번 구축한 후, 활성화된 Tier를 순서대로 실행합니다. 각 Tier는 이전 Tier가 탐지한 영역 정보를 받아 중복 작업을 피합니다.

```
입력 FASTA
    │
    ▼
BWTCore FM-index 구축 (suffix array, BWT, occurrence array)
    │
    ├─► Tier 1: 짧은 완벽 반복 (1–9 bp)
    │       │
    ├─► Tier 2: 중간/불완전 반복 (≥10 bp)
    │       │   [Tier 1 결과 영역 제외]
    │
    ├─► Tier 3: 긴 반복 (100 bp – 100 kbp)
    │       [Tier 1 + Tier 2 결과 영역 제외]
    │
    ▼
후처리: 정렬 → 인접 병합 → 겹침 필터 → 길이 필터
    │
    ▼
출력 (BED / VCF / TRF / STRfinder)
```

### Tier 1: 짧은 완벽 반복 (1–9 bp)

**담당 범위**: 모티프 길이 1–9 bp, 3 복사 이상의 완벽(또는 근사) 반복.

**동작 방식**:
- 서열 길이 < 10 Mbp: FM-index backward search로 모든 정규 모티프를 열거하고 직렬 반복 위치를 찾습니다.
- 서열 길이 ≥ 10 Mbp: 적응형 스텝 크기의 슬라이딩 윈도우 스캐너로 전환합니다.

STR(Short Tandem Repeat, 마이크로새틀라이트)의 주된 탐지 계층입니다. 8-mer 해시를 이용해 O(1) 단문 k-mer 조회를 수행합니다.

### Tier 2: 중간/불완전 반복 (≥10 bp)

**담당 범위**: 모티프 길이 10 bp 이상의 반복. 미니새틀라이트 및 중간 길이 불완전 반복을 탐지합니다.

**두 가지 서브 단계**:

1. **Long-unit strict** (단위 ≥20 bp): Kasai 알고리즘으로 구축한 LCP 배열에서 인접 접미사 쌍의 주기를 계산하고, 미스매치 허용 범위 내에서 확장합니다.

2. **General scanning** (단위 10–50 bp): BWT k-mer 시드 스캔(`bwt_seed.py`)으로 FM-index 발생 위치에서 주기적 수열(등차수열)을 탐지하고 후보를 확장합니다.

최대 20% 미스매치 및 10% 인델 허용.

### Tier 3: 긴 반복 (100 bp – 100 kbp)

**담당 범위**: 반복 단위 100 bp–100,000 bp. 위성 DNA, 센트로미어 반복, 트랜스포존 유래 반복 등을 탐지합니다.

**동작 방식**:
- BWT k-mer 시드 스캔 (큰 k-mer, 희박한 stride=100 기본값)
- 발생 위치에서 등차수열 탐지 후 후보 영역 결정
- **초장 어레이** (>100 복사 또는 >10 kb): 앵커 기반 경계 검증(`anchor_scan_boundaries`)으로 전체 DP 정렬 비용 절약
- **일반 어레이**: 전체 DP 정렬(`refine_repeat`)로 정밀 경계 결정

**적응형 파라미터**: 서열 길이, GC 함량, 기탐지 커버리지 비율에 따라 k-mer 크기, stride, max_occurrences 등을 자동 계산합니다 (아래 섹션 참조).

---

## 출력 포맷

### BED 형식 (기본값)

탭 구분 8열 형식. 좌표는 0-based(BED 표준).

```
chr1    100    145    AT    22.5    1    0.022    +
chr1    500    620    AATGG    24.0    2    0.083    +
chr1    1000   5400   ATCGATCG    550.0    3    0.150    -
```

| 열 | 설명 |
|----|------|
| 1 chrom | 염색체명 |
| 2 start | 시작 위치 (0-based) |
| 3 end | 종료 위치 |
| 4 motif | 반복 단위 서열 |
| 5 copies | 복사 수 |
| 6 tier | 탐지 Tier (1/2/3) |
| 7 mismatch_rate | 미스매치율 (0.0–1.0) |
| 8 strand | 가닥 (+/-) |

### VCF 형식

VCFv4.2 형식. 좌표는 1-based(VCF 표준). INFO 필드에 MOTIF, COPIES, TIER 등 포함.

```
##fileformat=VCFv4.2
##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the repeat">
#CHROM  POS   ID  REF  ALT   QUAL  FILTER  INFO
chr1    101   .   A    <STR>  .     .       END=145;MOTIF=AT;CONS_MOTIF=AT;COPIES=22.5;TIER=1;CONF=0.98;MM_RATE=0.022;...
```

### TRF .dat 형식

TRF(Tandem Repeats Finder)와 호환되는 형식. 공백 구분.

```
100 145 2 22.5 2 97 0 44 22 39 30 9 1.85 AT ATATATATATATATATATATAT...
```

열 순서: `Start End Period CopyNumber ConsensusSize PercentMatches PercentIndels Score A C G T Entropy ConsensusPattern Sequence`

### STRfinder .csv 형식

STRfinder 도구와 호환되는 CSV 형식. 좌표는 1-based.

```csv
STR_marker,STR_position,STR_motif,STR_genotype_structure,STR_genotype,STR_core_seq,Allele_coverage,Alleles_ratio,Reads_Distribution,STR_depth,Full_seq,Variations
STR_chr1_100,chr1:101-145,[AT]n,2[AT]22,22,ATATATATAT...,97%,-,22:22,22,ATATATATAT...,-
```

---

## Tier 3 Adaptive 파라미터

Tier 3는 `compute_adaptive_params()` 함수를 통해 입력 특성에 따라 탐색 파라미터를 자동으로 조정합니다.

### 입력 특성과 파라미터의 관계

| 입력 특성 | 영향을 받는 파라미터 | 효과 |
|-----------|---------------------|------|
| 서열 길이 (seq_len) | `kmer_size`, `stride`, `max_occurrences`, `scan_backward`, `scan_forward` | 길수록 k-mer 커짐, stride 커짐 |
| GC 함량 (gc_content) | `allowed_mismatch_rate` | 0.5에서 멀수록 미스매치 허용율 증가 |
| 기탐지 커버리지 비율 (coverage_ratio) | `stride`, `anchor_match_pct` | 커버리지 높을수록 stride 작아짐 (더 세밀) |
| 최대 주기 (max_period) | `tolerance_ratio` | 주기가 길수록 허용 오차 증가 |

### 파라미터 세부 설명

| 파라미터 | 범위 | 설명 |
|----------|------|------|
| `kmer_size` | 12–28 | FM-index 시드 k-mer 길이. 길수록 특이도 높음 |
| `stride` | 20–300 | k-mer 샘플링 간격. 클수록 빠르지만 민감도 낮음 |
| `allowed_mismatch_rate` | 0.15–0.20 | 허용 미스매치 비율 |
| `tolerance_ratio` | 0.02–0.04 | 주기 추정 오차 허용 비율 |
| `max_occurrences` | 200–1500 | k-mer 최대 발생 횟수 (빈번 k-mer 제외) |
| `anchor_match_pct` | 0.70–0.80 | 앵커 기반 경계 검증 최소 일치율 |
| `scan_backward` | 20–80 | 앵커에서 역방향 탐색 주기 수 |
| `scan_forward` | 200–800 | 앵커에서 순방향 탐색 주기 수 |

### 프리셋 설명

`--tier3-mode` 옵션은 speed_factor를 통해 속도 관련 파라미터를 비례적으로 조정합니다.

| 프리셋 | speed_weight | 특성 |
|--------|-------------|------|
| `fast` | 0.8 | k-mer 크기 증가, stride 증가, max_occurrences 감소. 대용량 게놈의 빠른 스크리닝에 적합 |
| `balanced` | 0.5 | 기본값. 속도와 민감도의 균형 |
| `sensitive` | 0.2 | k-mer 크기 감소, stride 감소, max_occurrences 증가. 짧은 서열이나 정밀 분석에 적합 |

### 서열 길이별 특수 처리

- **100 Mbp 초과 (대형 염색체 모드)**: `stride ≥ 150`, `kmer_size ≥ 20`, `max_occurrences ≤ 500`으로 강제 조정
- **100 kbp 미만 (미세 서열 모드)**: `stride ≥ 20`, `kmer_size ≥ 12`로 최소 민감도 보장

---

## 사용 예제

### 예제 1: 기본 실행

```bash
# 모든 Tier 실행, BED 출력 (기본값)
python3 -m src.main arabadopsis_chrs/chr1.fa -v

# 출력: chr1.bed
```

### 예제 2: 특정 Tier만 실행

```bash
# Tier 1만 실행 (STR/마이크로새틀라이트만)
python3 -m src.main input.fa --tiers tier1 --format bed -o str_only -v

# Tier 1과 Tier 2만 실행, VCF 출력
python3 -m src.main input.fa --tiers tier1,tier2 --format vcf -o output -v

# 주기 범위 제한 (1–50 bp만)
python3 -m src.main input.fa --tiers tier1,tier2 --min-period 1 --max-period 50 -o short_repeats
```

### 예제 3: Tier 3 fast 모드로 대용량 게놈 처리

```bash
# 대용량 게놈에서 긴 반복만 빠르게 탐색
python3 -m src.main large_genome.fa \
    --tiers tier3 \
    --tier3-mode fast \
    --min-period 100 \
    --min-array-bp 500 \
    --format bed \
    -o long_repeats \
    -v

# sensitive 모드로 소형 서열 정밀 분석
python3 -m src.main small_region.fa \
    --tier3-mode sensitive \
    --format trf \
    -o detailed_analysis
```

### 예제 4: 멀티 스레드 및 마스킹

```bash
# 4개 프로세스로 전체 게놈 병렬 처리
python3 -m src.main genome.fa -t 4 --format bed -o genome_repeats -v

# RepeatMasker 출력(소프트 마스크)에서 산재 반복 제외 후 분석
python3 -m src.main masked_genome.fa --mask soft -t 8 -v

# 하드 마스크(N 영역)만 제외하고 분석
python3 -m src.main genome.fa --mask hard -v

# 소프트+하드 마스크 모두 제외
python3 -m src.main masked_genome.fa --mask both -t $(nproc) -v
```

### 예제 5: 프로파일링으로 성능 분석

```bash
# 실행 시간 프로파일링 (상위 20개 핫스팟 출력)
python3 -m src.main input.fa --tiers tier1 --format trf --profile -v

# 프로파일 결과는 input.tier2_profile.prof 파일에도 저장됨
# python -m pstats input.tier2_profile.prof 으로 추가 분석 가능
```

---

## 테스트 실행

```bash
# 전체 테스트 실행 (micromamba 환경 사용 시)
micromamba run -n bwtandem python3 -m pytest tests/ -v

# 또는 환경 활성화 후 직접 실행
micromamba activate bwtandem
pytest tests/ -v

# 특정 테스트 파일 실행
pytest tests/test_adaptive_params.py -v   # Tier 3 적응형 파라미터
pytest tests/test_anchor_scan.py -v       # 앵커 기반 경계 스캔
pytest tests/test_tier3_wiring.py -v      # Tier 3 통합 연결
pytest tests/test_ground_truth.py -v -s   # 회귀 테스트 (상세 출력)
```

### 테스트 구성 (29개 테스트 + 스트레스 테스트)

| 테스트 파일 | 테스트 수 | 설명 | Cython 필요 |
|------------|----------|------|------------|
| `test_adaptive_params.py` | 10 | `compute_adaptive_params()` 단위 테스트: 프리셋별 동작, GC/커버리지 영향, 서열 길이별 모드 전환 | 아니오 |
| `test_anchor_scan.py` | 5 | `anchor_scan_boundaries()` 단위 테스트: 완벽 반복, 플랭킹 서열, 불완전 반복, 단일 복사본 | 아니오 |
| `test_tier3_wiring.py` | 3 | Tier 3 mode 파라미터 전달 검증: 초기화, 기본값, 시드 스캔 연결 | 아니오 |
| `test_ground_truth.py` | 11 | 합성 시퀀스 기반 회귀 테스트: Tier별 민감도/정밀도 검증 | Tier 2/3 테스트만 |
| `test_random_stress.py` | — | 30개 랜덤 시퀀스 스트레스 테스트 (`python3 -m tests.test_random_stress`로 실행) | 예 |

### 회귀 테스트 (Ground Truth)

`tests/fixtures/` 디렉터리에 100KB 합성 시퀀스(FASTA)와 정답 파일(BED)이 포함되어 있습니다. 각 시퀀스에는 알려진 위치에 알려진 모티프의 반복이 삽입되어 있으며, 테스트는 도구의 탐지 결과를 정답과 비교하여 민감도(sensitivity)와 정밀도(precision)를 자동 계산합니다.

#### 합성 테스트 데이터

| 파일 | 반복 수 | 설명 |
|------|---------|------|
| `synth_tier1.fa` / `synth_tier1_truth.bed` | 8 | Tier 1 반복: 1–6 bp 모티프, 완벽~5% 미스매치 |
| `synth_tier2.fa` / `synth_tier2_truth.bed` | 8 | Tier 2 반복: 12–50 bp 모티프, 완벽~8% 미스매치+인델 |
| `synth_tier3.fa` / `synth_tier3_truth.bed` | 8 | Tier 3 반복: 100–1000 bp 모티프, 완벽~10% 발산 |
| `synth_mixed.fa` / `synth_mixed_truth.bed` | 9 | 모든 Tier 혼합 (단일 서열에 Tier 1/2/3 반복 공존) |
| `synth_adjacent.fa` / `synth_adjacent_truth.bed` | 11 | 에지 케이스: 인접 반복, 접촉 반복, 복합 반복 |

합성 데이터 재생성:
```bash
python3 tests/fixtures/generate_synthetic.py
# seed=42로 재현 가능
```

#### 매칭 로직

정답과 예측 결과를 매칭할 때 다음 기준을 적용합니다:
1. **겹침 비율 (overlap ratio)** ≥ 50%: 두 구간의 겹침 길이 / 큰 구간의 길이
2. **모티프 호환성**: 정규 모티프(canonical motif)가 일치하거나, 기본 주기(primitive period) 길이가 호환 (정수배 또는 ±20% 이내)

불완전한 반복에서는 도구가 원래 모티프와 다른 컨센서스 모티프를 보고할 수 있으므로, 주기 호환성 검사를 통해 올바르게 탐지된 반복이 거짓 음성으로 분류되는 것을 방지합니다.

#### 테스트 결과 (2026-03-30)

**회귀 테스트 (합성 시퀀스, 11개 테스트)**

| 테스트 | 민감도 | 정밀도 | F1 | 임계값 |
|--------|--------|--------|-----|--------|
| **Tier 1** | 100.0% | 100.0% | 100.0% | 민감도 ≥ 95%, 정밀도 ≥ 90% |
| **Tier 2** | 100.0% | 100.0% | 100.0% | 민감도 ≥ 90%, 정밀도 ≥ 90% |
| **Tier 3** | 100.0% | 100.0% | 100.0% | 민감도 ≥ 90%, 정밀도 ≥ 90% |
| **Mixed (Tier 1)** | 100.0% | — | — | 민감도 ≥ 95% |
| **Mixed (Tier 2)** | 100.0% | — | — | 민감도 ≥ 70% |
| **Mixed (Tier 3)** | 100.0% | — | — | 민감도 ≥ 70% |
| **Adjacent** | 100.0% | 100.0% | 100.0% | 민감도 ≥ 95%, 정밀도 ≥ 90% |

**스트레스 테스트 (랜덤 30개 시퀀스, 108개 반복)**

| 항목 | 민감도 | 정밀도 | F1 | 정답 수 |
|------|--------|--------|-----|---------|
| **전체** | 100.0% | 93.1% | 96.4% | 108 |
| **Tier 1** | 100.0% | — | — | 54 |
| **Tier 2** | 100.0% | — | — | 31 |
| **Tier 3** | 100.0% | — | — | 23 |

스트레스 테스트는 30개의 50KB 랜덤 시퀀스에 2–5개의 반복을 삽입하여 탐지기의 강건성을 검증합니다. 재현 가능한 시드(1000–1029)를 사용합니다.

> **참고**: Tier 2/3/Mixed/Adjacent 테스트는 Cython 확장(`_accelerators.so`)이 빌드되어 있어야 실행됩니다. Cython 없이 실행하면 해당 테스트는 자동으로 건너뜁니다.

### 테스트 데이터

`arabadopsis_chrs/` 디렉터리에 Arabidopsis 염색체 FASTA 파일과 소형 테스트 서열(`test_seq1.fa`~`test_seq5.fa`)이 포함되어 있습니다.

```bash
# 테스트 서열로 빠른 동작 확인
python3 -m src.main arabadopsis_chrs/test_seq1.fa -v
```

### 유틸리티 스크립트

```bash
# 미스매치 허용 테스트용 변이 서열 생성
python3 scripts/mutate_fasta.py input.fa --mutation-rate 0.05
# input.fa.bak 백업 파일이 생성됨

# TRF 결과를 BED 형식으로 변환
python3 scripts/trf_to_bed.py input.dat -o output.bed
```

---

## 프로젝트 구조

```
bwtandem/
├── src/
│   ├── main.py             # CLI 진입점, FASTA 파싱, 출력 쓰기
│   ├── finder.py           # TandemRepeatFinder: 3-Tier 파이프라인 코디네이터
│   ├── bwt_core.py         # BWTCore: FM-index 구축 (접미사 배열, BWT, occurrence 배열)
│   ├── bwt_seed.py         # 공유 BWT k-mer 시드 스캔 (Tier 2/3 공용)
│   ├── tier1.py            # Tier1STRFinder: 짧은 완벽 반복 (1–9 bp)
│   ├── tier2.py            # Tier2LCPFinder: 중간/불완전 반복 (≥10 bp)
│   ├── tier3.py            # Tier3LongReadFinder: 긴 반복 + adaptive 파라미터
│   ├── motif_utils.py      # MotifUtils: 정규 모티프, 기본 주기 탐지, DP 정렬, 통계
│   ├── models.py           # 데이터 클래스: TandemRepeat, RefinedRepeat 등 + 출력 포맷터
│   ├── accelerators.py     # Cython 확장 투명 로더 (없으면 Python 폴백 사용)
│   ├── _accelerators.pyx   # Cython 소스: Hamming 거리, LCP 탐지, DP 정렬 등
│   └── utils.py            # 공통 유틸리티
├── tests/
│   ├── test_adaptive_params.py  # Tier 3 adaptive 파라미터 단위 테스트 (10개)
│   ├── test_anchor_scan.py      # 앵커 기반 경계 검증 테스트 (5개)
│   ├── test_tier3_wiring.py     # Tier 3 통합 연결 테스트 (3개)
│   ├── test_ground_truth.py     # 합성 시퀀스 회귀 테스트 (11개)
│   ├── test_random_stress.py    # 랜덤 30개 시퀀스 스트레스 테스트
│   └── fixtures/
│         ├── generate_synthetic.py    # 합성 데이터 생성 스크립트 (seed=42)
│         ├── synth_tier1.fa           # Tier 1 합성 시퀀스 (100KB)
│         ├── synth_tier1_truth.bed    # Tier 1 정답 BED
│         ├── synth_tier2.fa           # Tier 2 합성 시퀀스 (100KB)
│         ├── synth_tier2_truth.bed    # Tier 2 정답 BED
│         ├── synth_tier3.fa           # Tier 3 합성 시퀀스 (100KB)
│         ├── synth_tier3_truth.bed    # Tier 3 정답 BED
│         ├── synth_mixed.fa           # 혼합 Tier 합성 시퀀스 (100KB)
│         ├── synth_mixed_truth.bed    # 혼합 Tier 정답 BED
│         ├── synth_adjacent.fa        # 인접 반복 에지 케이스 (100KB)
│         └── synth_adjacent_truth.bed # 인접 반복 정답 BED
├── scripts/
│   ├── mutate_fasta.py     # 랜덤 점 변이 도입 (미스매치 허용 테스트용)
│   ├── run_trf.py          # TRF 실행 래퍼
│   └── trf_to_bed.py       # TRF .dat → BED 변환기
├── arabadopsis_chrs/       # Arabidopsis 테스트 데이터 (FASTA)
├── docs/                   # 추가 문서
├── results/                # 분석 결과 저장 디렉터리
├── Singularity             # Singularity 컨테이너 정의 파일
└── CLAUDE.md               # 개발 가이드
```

### 핵심 모듈 간 의존 관계

```
main.py
  └── finder.py (TandemRepeatFinder)
        ├── bwt_core.py (BWTCore)         ← 모든 Tier가 공유하는 FM-index
        ├── tier1.py
        ├── tier2.py
        │     └── bwt_seed.py
        └── tier3.py
              ├── bwt_seed.py             ← Tier 2와 공유
              └── accelerators.py         ← Cython 가속 (선택적)
```

---

## 설계 원칙

- **내부 좌표**: 0-based. VCF 및 STRfinder 출력 시 1-based로 변환
- **모티프 정규화**: 양쪽 가닥(forward + reverse complement)의 모든 순환 회전 중 사전 순으로 가장 작은 것을 정규 모티프로 사용
- **기본 주기 축약**: `refine_repeat()`는 항상 기본 주기로 축약 (예: ATAT → AT). 정확 및 근사(≤2% 오차) 주기성 검사 수행
- **센티넬 문자**: BWT 구축을 위해 서열 끝에 `$`를 추가하며, 반복 탐지에서는 제외
