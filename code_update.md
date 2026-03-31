# Code Update Log

## Step 1: Verify Code Functionality (Complete)
- All 11 existing tests passed
- Baseline time: 57 seconds (5 synthetic sequences of 100KB each)

## Step 2: Tier 1 Sliding Window Conversion (Complete)
- **Change**: 4^k motif enumeration -> O(n) sliding window scan
- **Result**: Test time reduced from 57s to 15s (3.7x improvement)
- **Accuracy**: 11/11 tests passed, stress test 100% sensitivity 99.1% precision
- **Key modifications**:
  - Direct detection of period-k repeats using `text[i] == text[i+k]` comparison
  - Allow short seeds (2 copies) -> extend with extend_with_mismatches then apply threshold
  - Fall back to seed region when extension fails

## Step 3: C Extension to Accelerate Tier 1 Run Detection (Complete)
- `src/c_extensions/tier1_scan.c`: Implemented period-k run detection in C
- Python fallback maintained (automatic switch if C build fails)
- 11/11 tests passed, stress test 100% sensitivity / 99.1% precision

## Step 4: bwtandem vs TRF Comparison (Complete)
- Comparison on 5 synthetic sequences (44 repeats):
  - bwtandem: 100% sensitivity, 97.8% precision
  - TRF: 97.7% sensitivity, 100% precision
- bwtandem outperformed TRF on adjacent repeats (11/11 vs 10/11)

## Step 5: Chr4 Execution (Complete)
- 18.5MB sequence, total runtime 7 min 52 sec (472 sec)
- BWT construction: 172s, Tier 1: 11s, Tier 2: 245s, Tier 3: 40s
- Repeats found: 6,625
- Tier 1 speed: previous 10-15 min -> 11s (~80x improvement)

## Step 6: Tool Comparison (Complete)

### Synthetic Sequence Comparison (44 ground truth repeats)
| Tool | Sensitivity | Precision | F1 |
|------|-------------|-----------|-----|
| bwtandem | 100.0% | 97.8% | 98.9% |
| TRF | 97.7% | 100.0% | 98.9% |

### Chr4 Execution Comparison
| Tool | Results | Runtime |
|------|---------|---------|
| bwtandem | 6,625 | 7 min 52 sec |
| TRF | 4,549 | 34 sec |
| mreps | 84,502 | 48 sec |
| ULTRA | 23,145 | 24 min 11 sec |

### Conclusions
- bwtandem: Highest sensitivity (100%), detects 45% more repeats than TRF
- TRF: Fastest (34 sec), high precision
- mreps: Over-detection (84K), filtering required
- ULTRA: Slowest (24 min), 3.5x more results than bwtandem
