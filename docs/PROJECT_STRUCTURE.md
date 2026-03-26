**Project Structure**

- **Purpose:**: High-level guide to the repository layout, running the pipeline, and where to find important modules and data.

**Top-level layout:**

- **`arabadopsis_chrs/`**: Example or input FASTA files used for development and quick testing (`Chr1.fa`, `test_seq4.fa`, etc.).
- **`results/`**: Output files from runs (BED files, profiles, and other artifacts). Example files include `ChrC_tier2_twelfth.bed` and `test3.bed`.
- **`scripts/`**: Utility scripts (for example `mutate_fasta.py`).
- **`src/`**: Core source code. Key modules:
  - `src/main.py`: CLI entry point and pipeline orchestration.
  - `src/bwt_core.py`: BWT / LCP algorithms and primitives used by Tier 2.
  - `src/tier1.py`, `src/tier2.py`, `src/tier3.py`: The pipeline tiers (see `docs/tier1.md`, `docs/tier2.md`, `docs/tier3.md`).
  - `src/accelerators.py` and `_accelerators.*`: Performance-oriented implementations (C / Cython / Pyx bindings) called from Python for heavy loops.
  - `src/models.py`, `src/utils.py`: Data shapes, helpers, and common utilities.

**How to run (examples):**

Run Tier 2 only (verbose recommended for debugging):

```
PYTHONUNBUFFERED=1 python3 -m src.main arabadopsis_chrs/test_seq4.fa --tiers tier2 --output results/test2.bed --format bed --verbose
```

Run the full pipeline (all tiers):

```
python3 -m src.main arabadopsis_chrs/test_seq4.fa --tiers tier1,tier2,tier3 --output results/final.bed --format bed
```

**Dependencies & build notes:**

- The project runs on Python 3.8+ (verify locally); if a Cython/C extension is used, compile it before large runs. See `_accelerators.*` and `src/_accelerators.pyx`.
- If you rely on compiled accelerators, run the build steps (typical pattern):

```
python3 -m pip install -r requirements.txt
python3 setup.py build_ext --inplace   # if a setup.py exists or follow your build instructions
```

**Outputs and results:**

- Standard output format is BED (`--format bed`) and output path is controlled with `--output`.
- Example output files live in `results/` — use them as references for expected fields and formatting.

**Debugging workflow:**

- Start with a small FASTA from `arabadopsis_chrs/` and run one tier at a time with `--verbose`.
- Compare intermediate outputs between tiers to verify data flow.

**Where to make changes:**

- Algorithmic work: `src/bwt_core.py`, `src/tier*.py`.
- Performance/optimizations: `src/accelerators.py`, `_accelerators.c`, `src/_accelerators.pyx`.
- CLI / orchestration: `src/main.py`.

**Next steps (suggested improvements):**

- Add a top-level `README.md` describing quickstart and developer setup.
- Add `requirements.txt` or `pyproject.toml` if not present, to pin dependencies.
- Add unit tests for small deterministic functions in each `src/` module.
