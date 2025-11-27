# Tier2 Tandem Repeat Search: Detailed Explanation

## Overview

Tier2 in this algorithm is designed to find **imperfect tandem repeats** with motif lengths greater than 9bp (i.e., not classic microsatellites). It uses a combination of BWT/FM-index, LCP arrays, and seed-and-extend strategies with mismatch/indel tolerance. The process is optimized for large genomes and can be accelerated with Cython.

## Main Steps

### 1. Initialization
- The input sequence is converted to uppercase and a sentinel `$` is appended if missing.
- A Burrows-Wheeler Transform (BWT) and FM-index are built for the sequence.
- Tier2 is initialized with parameters:
  - `min_period` (internally forced to ≥10bp)
  - `max_period` (user-defined)
  - `allowed_mismatch_rate` and `allowed_indel_rate` (default: 0.2, 0.1)

### 2. Long Unit Repeat Search (`find_long_unit_repeats_strict`)
- Scans for long, adjacent repeats (default: 20–120bp units).
- For each candidate unit length (longest first):
  - Slide a window across the sequence.
  - For each window, compare adjacent units using Hamming distance and allow small indels (shifts).
  - If enough adjacent copies are found, reduce to primitive period (e.g., 105bp → 35bp if periodic).
  - Refine and record the repeat.

**Example:**
- Sequence: `ATGATGATGATGATG...` (motif: ATG, 4 copies)
- For unit_len=3, finds the repeat, checks for mismatches/indels, and records start/end/copies.

### 3. General Scanning for Medium/Long Repeats (`find_long_repeats` → `_find_repeats_simple`)
- Scans the genome for repeats with motif lengths from 10bp up to `max_period`.
- Uses adaptive step sizes (`position_step`, `period_step`) for speed.
- For each period:
  - Slide a window across the sequence.
  - For each window:
    - Check if the window and its offset by `period` match (quick filter).
    - If so, use `_extend_with_mismatches` to grow the repeat left/right, allowing mismatches/indels.
    - If enough copies are found, refine and record the repeat.
- Progress is reported every 2 seconds (period change) and every 60 seconds (within a period).

**Example:**
- Sequence: `ACGTACGTACGTACGT...` (motif: ACGT, 4 copies)
- For period=4, window at i=0 matches i+4, extension finds 4 copies, records repeat.
- If mismatches are present (e.g., `ACGTACGAACGT...`), extension allows up to the mismatch threshold.

### 4. Post-processing
- All repeats are sorted by position.
- Adjacent repeats with the same motif are merged.
- Overlapping repeats are filtered, keeping the longest/best.
- Only repeats within user-specified bounds are kept.

## Cython Acceleration
- The `_extend_with_mismatches` function is performance-critical and can use a Cython implementation for speed.
- The Cython code efficiently extends repeats, computes mismatches, and handles partial extensions.

## Example Workflow

Suppose you run:
```bash
python -m src.main arabadopsis_chrs/ChrC.fa --tiers tier2 --output results/ChrC_tier2 --format bed --verbose
```
- The algorithm builds the BWT/FM-index.
- Tier2 scans for motif lengths 10–50bp (default), skipping 1–9bp.
- For each motif length, it slides windows, checks for periodicity, and extends repeats with allowed mismatches.
- Progress is printed every 2 seconds (period change) and every 60 seconds (within a period).
- Results are written in BED format.

## Example Output (BED)
```
ChrC    1000    1120    ATG    40    +
ChrC    5000    5200    ACGTACGT    25    -
```

## Notes
- Tier2 is designed for larger, imperfect repeats and will not re-find classic microsatellites (1–9bp).
- Sensitivity and speed can be tuned by adjusting step sizes, mismatch/indel rates, and period bounds.
- For perfect short repeats, use Tier1.

---
For more details, see the source files: `src/tier2.py`, `src/finder.py`, and `src/_accelerators.pyx`.
