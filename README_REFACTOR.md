# Refactored BWT-based Tandem Repeat Finder

This project has been refactored from a monolithic script into a modular Python package.

## Structure

- `src/`: Source code package
  - `main.py`: CLI entry point (replaces `bwt.py`)
  - `finder.py`: Main coordinator class `TandemRepeatFinder`
  - `bwt_core.py`: Core BWT and FM-index implementation
  - `tier1.py`: Tier 1 finder (Short Perfect Repeats)
  - `tier2.py`: Tier 2 finder (Imperfect & Medium Repeats)
  - `tier3.py`: Tier 3 finder (Long Reads)
  - `models.py`: Data classes (`TandemRepeat`, `AlignmentResult`)
  - `motif_utils.py`: Motif manipulation and alignment utilities
  - `utils.py`: General helper functions

## Usage

Run the tool using the `src.main` module:

```bash
python3 -m src.main input.fasta --output results --format bed
```

### Options

- `--min-period`: Minimum period size (default: 1)
- `--max-period`: Maximum period size (default: 2000)
- `--output`, `-o`: Output file prefix
- `--format`: Output format (`bed`, `vcf`, `trf`, `strfinder`)
- `--verbose`, `-v`: Show progress

## Testing

Run the tests using `unittest`:

```bash
python3 tests/test_refactored.py
```

## Improvements

- **Modularity**: Code is split into logical modules for better maintainability.
- **Efficiency**: Adaptive scanning and optimized loops for large sequences.
- **Accuracy**: Improved variation detection and consensus calling.
- **Type Safety**: Added type hints throughout the codebase.
