# BWTandem

A BWT-based Tandem Repeat Finder for genomic sequences. This tool uses the Burrows-Wheeler Transform (BWT) and FM-index to efficiently detect tandem repeats in DNA sequences.

## Installation

### Requirements

- Python 3.8+
- NumPy

### Optional Dependencies (for acceleration)

- `pydivsufsort` - Fast C-based suffix array construction
- `numba` - JIT compilation for performance-critical functions
- Cython - For native accelerator module (`_accelerators.pyx`)

### Install

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd bwtandem
pip install numpy
```

For optimal performance, install optional dependencies:

```bash
pip install pydivsufsort numba
```

## Usage

### Basic Usage

```bash
python -m src.main <fasta_file>
```

### Command Line Options

```
usage: main.py [-h] [--min-period MIN_PERIOD] [--max-period MAX_PERIOD]
               [--min-array-bp MIN_ARRAY_BP] [--max-array-bp MAX_ARRAY_BP]
               [--tiers TIERS] [--output OUTPUT] [--format {bed,vcf,trf,strfinder}]
               [--verbose] [--profile]
               fasta_file

BWT-based Tandem Repeat Finder

positional arguments:
  fasta_file            Input FASTA file

optional arguments:
  -h, --help            show this help message and exit
  --min-period MIN_PERIOD
                        Minimum period size (default: 1)
  --max-period MAX_PERIOD
                        Maximum period size (default: 2000)
  --min-array-bp MIN_ARRAY_BP
                        Minimum repeat array length in bp (default: no minimum)
  --max-array-bp MAX_ARRAY_BP
                        Maximum repeat array length in bp (default: no maximum)
  --tiers TIERS         Comma-separated list of tiers to run (tier1,tier2,tier3) or 'all'
                        (default: tier1,tier2,tier3)
  --output, -o OUTPUT   Output file prefix (default: input filename)
  --format {bed,vcf,trf,strfinder}
                        Output format (default: bed)
  --verbose, -v         Show progress
  --profile             Profile execution with cProfile and print top hotspots
```

### Examples

Find all tandem repeats in a FASTA file:

```bash
python -m src.main genome.fa
```

Find repeats with motifs between 2-50bp, with verbose output:

```bash
python -m src.main genome.fa --min-period 2 --max-period 50 -v
```

Run only Tier 1 (short perfect repeats) and output in VCF format:

```bash
python -m src.main genome.fa --tiers tier1 --format vcf -o results
```

## Output Formats

### BED Format (default)

Tab-separated format with columns:
- Chromosome
- Start position (0-based)
- End position
- Consensus motif
- Copy number
- Tier (1, 2, or 3)
- Mismatch rate
- Strand

### VCF Format

Standard VCF 4.2 format with INFO fields containing repeat details.

### TRF Format

TRF-compatible DAT format for compatibility with downstream tools.

### STRfinder Format

CSV format compatible with STRfinder output specifications.

## Detection Tiers

BWTandem uses a multi-tier approach for comprehensive repeat detection:

- **Tier 1**: Short perfect tandem repeats (1-9bp motifs). Uses optimized sliding window with adaptive sampling.
- **Tier 2**: Imperfect and medium-length repeats (≥10bp motifs). Uses BWT/FM-index with LCP arrays and seed-and-extend with mismatch tolerance.
- **Tier 3**: Long-read repeats (100bp+ motifs). Uses heuristic k-mer sampling and FM-index for detecting very long repeats with significant variation.

## Project Structure

```
bwtandem/
├── src/
│   ├── main.py           # CLI entry point
│   ├── finder.py         # Main TandemRepeatFinder coordinator
│   ├── bwt_core.py       # BWT construction and FM-index operations
│   ├── tier1.py          # Tier 1: Short perfect repeat finder
│   ├── tier2.py          # Tier 2: Imperfect/medium repeat finder
│   ├── tier3.py          # Tier 3: Long-read repeat finder
│   ├── models.py         # Data models (TandemRepeat, etc.)
│   ├── motif_utils.py    # Motif handling utilities
│   ├── accelerators.py   # Optional Cython accelerators
│   ├── _accelerators.pyx # Cython extension source
│   └── utils.py          # General utilities
├── scripts/
│   └── mutate_fasta.py   # Utility script for mutating FASTA files
└── results/              # Sample output files and test sequences
```

## License

This project is open source.
