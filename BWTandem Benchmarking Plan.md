---
title: BWTandem Benchmarking Plan

---

# BWTandem Benchmarking Plan

## Overview

This document describes three benchmarking experiments to evaluate BWTandem against three existing tandem repeat detection tools: TRF, mreps, and ULTRA. Each experiment uses a different genome and focuses on a different aspect of tandem repeat detection.

**Tools to compare (8 total):**

| Tool | Description | Source | Installation |
|------|------------|--------|-------------|
| BWTandem | BWT-based multi-tier TR finder | `https://github.com/framazan/bwtandem` | Follow repo instructions |
| TRF | k-tuple matching + WDP alignment (Benson, 1999) | `https://github.com/Benson-Genomics-Lab/TRF` | `conda install -c bioconda trf` |
| mreps | Combinatorial string repetition (Kolpakov et al., 2003) | `https://github.com/gregorykucherov/mreps` | `conda install -c bioconda mreps` |
| ULTRA | HMM Viterbi-based detection (Olson & Wheeler, 2024) | `https://github.com/TravisWheelerLab/ULTRA` | `conda install -c bioconda ultra` |
| TideHunter | Seed-and-chain for noisy long reads (Gao et al., 2019) | `https://github.com/yangao07/TideHunter` | `conda install -c bioconda tidehunter` |
| tantan | HMM-based repeat masking (Frith, 2011) | `https://gitlab.com/mcfrith/tantan` | `conda install -c bioconda tantan` |
| NCRF | Motif-guided TR detection in noisy reads (Harris et al., 2019) | `https://github.com/makovalab-psu/NoiseCancellingRepeatFinder` | `conda install -c bioconda ncrf` |
| TRASH | De novo TR annotation with HOR detection (Wlodzimierz et al., 2023) | `https://github.com/vlothec/TRASH` | See GitHub README (requires R) |

**Important tool characteristics to note:**
- **TideHunter** was designed for noisy long reads (PacBio/ONT), not assembled genomes. It works on FASTA input but may behave differently on high-quality assemblies vs raw reads. No maximum period limitation.
- **tantan** is primarily a repeat masking tool (outputs lowercase-masked sequence), not a repeat finder. It identifies low-complexity and short-period tandem repeats for masking. To get BED-like output, use `tantan -f4` which outputs repeat coordinates.
- **NCRF** requires the user to specify known motifs as input. It does NOT perform de novo repeat discovery. For benchmarking, you need to provide the motif sequences (e.g., CEN180, CentC, knob180, TAG) explicitly. This makes it useful for targeted detection experiments (Experiments 2 and 3) but not ideal for Experiment 1.
- **TRASH** specializes in centromeric satellite annotation with higher-order repeat (HOR) detection. It uses local k-mer counting and is particularly strong for Experiment 2 (Col-CEN). It also outputs GFF format, not BED.

---

## Experiment 1: General Tandem Repeat Detection (GIAB adotto benchmark)

### Purpose
Compare how many tandem repeat regions each tool detects against a curated human tandem repeat catalog. This tests overall detection coverage across all repeat types.

### Data

**Ground truth:** GIAB Tandem Repeat Benchmark from the adotto project
- Repository: `https://github.com/ACEnglish/adotto`
- Publication: English et al. "Analysis and benchmarking of small and large genomic variants across tandem repeats." *Nature Biotechnology* (2024). https://doi.org/10.1038/s41587-024-02225-z
- The benchmark BED file (`tr_regions.bed.gz`) contains curated tandem repeat regions across the human genome (GRCh38).
- Download the latest release from: `https://github.com/ACEnglish/adotto/releases`

**Reference genome:** GRCh38 (human)
- Download from NCBI or UCSC
- Note: For computational feasibility, you may start with a single chromosome (e.g., chr1 or chr21) before running the full genome.

### Procedure

1. Download the adotto benchmark BED file and the GRCh38 reference FASTA.
2. Run all four tools on the same reference FASTA (or selected chromosome).
3. Convert each tool's output to BED format (chrom, start, end).
4. Use `bedtools intersect` to compare each tool's output against the adotto benchmark BED.
5. Calculate:
   - **Total regions detected** by each tool
   - **Overlap with benchmark**: number and percentage of adotto regions that overlap with each tool's calls
   - **Unique detections**: regions found by one tool but not others
   - **Runtime and peak memory** for each tool

### Key commands (example)

```bash
# Run TRF
- [x] trf genome.fa 2 7 7 80 10 50 2000 -d -h

# Run mreps
- [x] mreps -res 5 -from 1 -to [seqlen] genome.fa

# Run ULTRA
- [x] ULTRA --max_period 2000 genome.fa output.bed

# Run BWTandem
- [x] python -m src.main genome.fa --format bed --output bwtandem_output

# Run TideHunter (designed for reads, but works on assembled FASTA)
TideHunter -f 2 genome.fa > tidehunter_output.txt # nope, we can't do this

# Run tantan (masking mode, output coordinates with -f4)
- [x] tantan -f4 genome.fa > tantan_output.txt

# Run NCRF (requires specifying motifs -- NOT suitable for de novo Experiment 1)
# Skip NCRF for this experiment or provide a comprehensive motif list

# Run TRASH (de novo mode)
- [x] bash TRASH_run.sh genome.fa --def

# Compare with benchmark
bedtools intersect -a adotto_tr_regions.bed -b tool_output.bed -wa -u | wc -l # NEXT STEP
```

**Note on NCRF:** Since NCRF requires known motifs as input, it is not well-suited for Experiment 1 (general de novo detection). Include NCRF results only if you provide a comprehensive motif list derived from the adotto catalog. Otherwise, mark NCRF as "N/A - requires known motifs" in the results table.

### Expected output
A summary table like:

| Tool | Total regions | Overlap with adotto (%) | Unique regions | Runtime | Memory |
|------|-------------|----------------------|---------------|---------|--------|
| BWTandem | ? | ? | ? | ? | ? |
| TRF | ? | ? | ? | ? | ? |
| mreps | ? | ? | ? | ? | ? |
| ULTRA | ? | ? | ? | ? | ? |
| TideHunter | ? | ? | ? | ? | ? |
| tantan | ? | ? | ? | ? | ? |
| NCRF | N/A (motif-guided) | ? | ? | ? | ? |
| TRASH | ? | ? | ? | ? | ? |

---

## Experiment 2: Centromere Detection in Arabidopsis (Col-CEN)

### Purpose
Test how well each tool detects the CEN180 centromeric satellite repeat (~178 bp monomer) in the Arabidopsis T2T genome. The centromeres are megabase-scale tandem arrays of CEN180 repeats. This is a focused test on large satellite arrays -- NOT a genome-wide general repeat comparison. Only evaluate centromeric repeat detection.

### Data

**Genome:** Col-CEN v1.2 Arabidopsis thaliana T2T assembly
- Publication: Naish et al. "The genetic and epigenetic landscape of the Arabidopsis centromeres." *Science* 374, 1326-1335 (2021). https://doi.org/10.1126/science.abi7489
- Assembly download: `https://github.com/schatzlab/Col-CEN`
- NCBI accession: GCA_028009825.2 (or search "Col-CEN" on NCBI Assembly)
- The assembly contains 5 pseudomolecules with resolved centromeres on all chromosomes.

**Known centromere positions:**
The centromeres are characterized by CEN180 satellite repeat arrays. Approximate centromere positions from the Col-CEN paper (Table S1 / Figure 1):

| Chromosome | Centromere approximate region |
|-----------|------------------------------|
| Chr1 | ~14.5-17.5 Mb |
| Chr2 | ~3.5-6.0 Mb |
| Chr3 | ~13.0-16.5 Mb |
| Chr4 | ~3.0-6.5 Mb |
| Chr5 | ~11.5-15.5 Mb |

Note: Get precise coordinates from the Col-CEN supplementary data or annotation files in the GitHub repo. You can also use the CEN180 monomer consensus sequence (178 bp) from the paper to create a reference BED of CEN180 locations using BLAST or similar.

### Procedure

1. Download the Col-CEN assembly FASTA.
2. Run all four tools on the full assembly (Arabidopsis is small, ~134 Mb, so this is computationally feasible).
3. For each tool's output, filter for repeats that:
   - Fall within the known centromere regions (use `bedtools intersect`)
   - Have a period/motif length near 178 bp (or multiples thereof)
4. Calculate:
   - **Centromere coverage**: what percentage of each centromere region is covered by the tool's repeat calls?
   - **CEN180 detection**: how many CEN180 monomers are detected?
   - **Fragmentation**: does the tool report the centromere as one large array or many small fragments?
   - **Runtime** on the full Arabidopsis genome

### What to look for
- TRF has a 2000 bp period limit, so it should still detect CEN180 (178 bp < 2000 bp), but may fragment large arrays.
- mreps cannot handle indels, so diverged CEN180 copies may be missed.
- ULTRA should handle composition bias well.
- BWTandem: this is where Tier 2 (long unit detection) and Tier 3 should contribute.
- TideHunter: designed for noisy reads but should detect tandem structure in assembly. May fragment or merge differently than reference-based tools.
- tantan: will mask CEN180 regions but may not report individual monomers. Compare the total masked bp within centromeres.
- NCRF: provide the CEN180 consensus (178 bp) as the motif input. This is an ideal use case for NCRF since the motif is known. Also provide CEN159 (159 bp) as a second motif. Example: `cat genome.fa | NCRF CEN180_consensus CEN159_consensus --minlength=500 > ncrf_output.txt` 
- TRASH: this is TRASH's strongest use case. It was benchmarked on Col-CEN in the original paper (Wlodzimierz et al. 2023). TRASH also detects higher-order repeats (HORs) within the satellite arrays, which is unique among these tools.
    * Run with CEN178 and CEN159 templates for guided mode
    * Run de novo for comparison.

### Expected output
A per-centromere summary:

| Centromere | Size (Mb) | BWTandem (%) | TRF (%) | mreps (%) | ULTRA (%) | TideHunter (%) | tantan (%) | NCRF (%) | TRASH (%) |
|-----------|----------|-------------|---------|----------|----------|---------------|-----------|---------|----------|
| CEN1 | ~3.0 | ? | ? | ? | ? | ? | ? | ? | ? |
| CEN2 | ~2.5 | ? | ? | ? | ? | ? | ? | ? | ? |
| CEN3 | ~3.5 | ? | ? | ? | ? | ? | ? | ? | ? |
| CEN4 | ~3.5 | ? | ? | ? | ? | ? | ? | ? | ? |
| CEN5 | ~4.0 | ? | ? | ? | ? | ? | ? | ? | ? |

---

## Experiment 3: Microsatellite, Satellite, and CentC Detection in Maize (T2T Mo17)

### Purpose
Test detection of three biologically important repeat classes in a large, complex plant genome: microsatellites (SSRs), satellite arrays (knob180, TR-1), and centromeric CentC repeats. The maize T2T Mo17 genome is ideal because it has fully resolved centromeres and extremely long satellite arrays.

### Data

**Genome:** T2T Mo17 maize assembly
- Publication: Chen et al. "A complete telomere-to-telomere assembly of the maize genome." *Nature Genetics* 55, 1221-1231 (2023). https://doi.org/10.1038/s41588-023-01419-6
- NCBI Assembly: GCA_022117705.1 (search "Zm-Mo17-REFERENCE-CAU-T2T-assembly")
- Genome size: 2,178.6 Mb (large -- plan accordingly for runtime)

**Known repeat features from the paper:**
- **Microsatellites**: 5.45 Mb total. Includes super-long TAG trinucleotide repeat arrays up to 235 kb.
- **Satellite arrays**: knob180 (25 arrays), TR-1 (17 arrays), and other satellites. Supplementary Tables 6-9 in the paper list precise locations and sizes.
- **CentC**: 17 arrays located in centromeric/pericentromeric regions. The CentC monomer is ~156 bp. Supplementary Table 8 has positions.
- **Other notable satellites**: Cent4 (260 kb), sat268 (176 kb), sat261 (152 kb), sat112 (78 kb).

**CentC consensus sequence** (~156 bp):
You can obtain the CentC consensus from the Mo17 paper supplementary materials or from MaizeGDB/NCBI (search "CentC maize repeat").

### Procedure

**Note:** The Mo17 genome is 2.18 Gb. Running all four tools on the full genome will be time-consuming. Consider:
- Starting with 1-2 chromosomes (e.g., chr6 for knob180, chr8 for CentC)
- Recording runtime carefully, as this is a key benchmark metric for large genomes

#### 3A. Microsatellite detection

1. Run all four tools with parameters targeting short repeats (period 1-6 bp).
2. Filter output for repeats with period <= 6 bp.
3. Compare:
   - Total microsatellite bp detected
   - Detection of the known TAG repeat arrays (the paper reports specific locations and sizes)
   - Can each tool find the 235 kb TAG array on chr6? Does it report it as one unit or fragments?

#### 3B. Satellite / knob detection

1. Run all four tools with parameters covering period ~180 bp (knob180) and ~358 bp (TR-1).
2. Cross-reference with the known array positions from Supplementary Tables 6-7.
3. Compare:
   - Number of knob180 arrays detected (out of 25 known)
   - Number of TR-1 arrays detected (out of 17 known)
   - Boundary accuracy: how close are the reported start/end positions to the known coordinates?

#### 3C. CentC detection

1. Run all four tools with parameters covering period ~156 bp.
2. Cross-reference with Supplementary Table 8 (17 known CentC arrays).
3. Compare:
   - Number of CentC arrays detected (out of 17 known)
   - Coverage of centromeric regions
   - Can each tool distinguish CentC-rich vs CentC-poor centromeres? (The Mo17 paper describes both types)

### Key parameters to use

```bash
# TRF - default handles up to 2000 bp period
trf chr6.fa 2 7 7 80 10 50 2000 -d -h

# mreps - set resolution high enough for diverged satellites
mreps -res 10 -minperiod 1 -maxperiod 500 chr6.fa

# ULTRA - adjust max period
ULTRA --max_period 500 chr6.fa chr6_ultra.bed

# BWTandem - full range
python -m src.main chr6.fa --min-period 1 --max-period 500 --format bed -o chr6_bwtandem

# TideHunter
TideHunter -f 2 -p 1 -P 500 chr6.fa > chr6_tidehunter.txt

# tantan - coordinate output mode
tantan -f4 chr6.fa > chr6_tantan.txt

# NCRF - provide specific motifs for targeted detection
cat chr6.fa | NCRF TAG CentC_consensus knob180_consensus --minlength=200 > chr6_ncrf.txt
# You need the actual consensus sequences for CentC (~156bp), knob180 (~180bp), TR-1 (~358bp)
# Get these from the Mo17 paper supplementary or MaizeGDB

# TRASH - de novo and/or with templates
bash TRASH_run.sh chr6.fa --def
# Or with templates:
bash TRASH_run.sh chr6.fa --templates CentC.fa,knob180.fa,TR1.fa --def
```

### Expected output

**3A. Microsatellite summary:**

| Tool | Total SSR bp | TAG arrays found | Longest TAG (kb) | Runtime |
|------|-------------|-----------------|------------------|---------|
| BWTandem | ? | ? | ? | ? |
| TRF | ? | ? | ? | ? |
| mreps | ? | ? | ? | ? |
| ULTRA | ? | ? | ? | ? |
| TideHunter | ? | ? | ? | ? |
| tantan | ? | ? | ? | ? |
| NCRF | ? | ? | ? | ? |
| TRASH | ? | ? | ? | ? |

**3B. Satellite summary:**

| Tool | knob180 arrays (of 25) | TR-1 arrays (of 17) | Mean boundary offset (bp) |
|------|----------------------|--------------------|--------------------------| 
| BWTandem | ? | ? | ? |
| TRF | ? | ? | ? |
| mreps | ? | ? | ? |
| ULTRA | ? | ? | ? |
| TideHunter | ? | ? | ? |
| tantan | ? | ? | ? |
| NCRF | ? | ? | ? |
| TRASH | ? | ? | ? |

**3C. CentC summary:**

| Tool | CentC arrays (of 17) | Total CentC bp | Centromere coverage (%) |
|------|---------------------|---------------|------------------------|
| BWTandem | ? | ? | ? |
| TRF | ? | ? | ? |
| mreps | ? | ? | ? |
| ULTRA | ? | ? | ? |
| TideHunter | ? | ? | ? |
| tantan | ? | ? | ? |
| NCRF | ? | ? | ? |
| TRASH | ? | ? | ? |

---

## General Notes

### Output format standardization
All tools produce different output formats. Convert everything to BED (chrom, start, end, motif, copies) for fair comparison. Write a conversion script for each tool:
- **TRF**: parse `.dat` file
- **mreps**: parse text output
- **ULTRA**: already outputs BED-like format
- **BWTandem**: use `--format bed`
- **TideHunter**: parse tabular output (columns: readName, repN, copyNum, readLen, start, end, consLen, aveMatch, fullLen, subPos)
- **tantan**: use `-f4` flag for BED-like coordinate output; otherwise parse lowercase-masked FASTA
- **NCRF**: parse alignment output using included `ncrf_to_bed.py` helper or write custom parser
- **TRASH**: parse GFF output file (`all.repeats.from.assembly.fa.csv`) or the GFF file

### Runtime measurement
Use `/usr/bin/time -v` (GNU time) to measure both wall clock time and peak RSS memory:
```bash
/usr/bin/time -v trf genome.fa 2 7 7 80 10 50 2000 -d -h 2> trf_time.log
```

### Computational resources
- Experiment 1 (human): expect 1-8 hours per tool depending on chromosome count. Use HPC.
- Experiment 2 (Arabidopsis): small genome (~134 Mb), should complete in minutes to 1 hour per tool on a single node.
- Experiment 3 (maize): large genome (~2.18 Gb). Plan for several hours per tool. Start with individual chromosomes.

### bedtools commands for comparison
```bash
# Install bedtools
conda install -c bioconda bedtools

# Sort BED files
sort -k1,1 -k2,2n tool_output.bed > tool_output.sorted.bed

# Intersect with benchmark/known regions
bedtools intersect -a known_regions.bed -b tool_output.sorted.bed -wa -u | wc -l

# Calculate coverage of known regions
bedtools coverage -a known_regions.bed -b tool_output.sorted.bed > coverage.txt

# Find regions unique to one tool
bedtools intersect -a tool1.bed -b tool2.bed -v > tool1_unique.bed
```

### References

1. Benson G. "Tandem repeats finder: a program to analyze DNA sequences." *Nucleic Acids Research* 27(2):573-580 (1999).
2. Kolpakov R, Bana G, Kucherov G. "mreps: Efficient and flexible detection of tandem repeats in DNA." *Nucleic Acids Research* 31(13):3672-3678 (2003).
3. Olson ND, Wheeler TJ. "ULTRA: ULTRA Locates Tandemly Repetitive Areas." *Bioinformatics Advances* 4(1):vbae149 (2024).
4. Gao Y et al. "TideHunter: efficient and sensitive tandem repeat detection from noisy long-reads using seed-and-chain." *Bioinformatics* 35(14):i200-i207 (2019). https://github.com/yangao07/TideHunter
5. Frith MC. "A new repeat-masking method enables specific detection of homologous sequences." *Nucleic Acids Research* 39(4):e23 (2011). https://gitlab.com/mcfrith/tantan
6. Harris RS, Cechova M, Makova KD. "Noise-cancelling repeat finder: uncovering tandem repeats in error-prone long-read sequencing data." *Bioinformatics* 35(22):4809-4811 (2019). https://github.com/makovalab-psu/NoiseCancellingRepeatFinder
7. Wlodzimierz P, Hong M, Henderson IR. "TRASH: Tandem Repeat Annotation and Structural Hierarchy." *Bioinformatics* 39(5):btad308 (2023). https://github.com/vlothec/TRASH
8. English AC et al. "Analysis and benchmarking of small and large genomic variants across tandem repeats." *Nature Biotechnology* (2024).
9. Naish M et al. "The genetic and epigenetic landscape of the Arabidopsis centromeres." *Science* 374:1326-1335 (2021).
10. Chen J et al. "A complete telomere-to-telomere assembly of the maize genome." *Nature Genetics* 55:1221-1231 (2023).