#!/usr/bin/env python3
"""Generate 5 synthetic FASTA + ground truth BED files for regression testing.

Each sequence ~100 KB with known tandem repeats at exact positions in random DNA.
Run this script once to create the fixture files.
"""
import random
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
random.seed(42)


def random_dna(length: int, gc: float = 0.45) -> str:
    bases = []
    for _ in range(length):
        r = random.random()
        if r < gc / 2:
            bases.append('G')
        elif r < gc:
            bases.append('C')
        elif r < gc + (1 - gc) / 2:
            bases.append('A')
        else:
            bases.append('T')
    return ''.join(bases)


def mutate(seq: str, rate: float) -> str:
    bases = list(seq)
    alts = {'A': 'TCG', 'T': 'ACG', 'C': 'ATG', 'G': 'ATC'}
    for i in range(len(bases)):
        if random.random() < rate:
            bases[i] = random.choice(alts[bases[i]])
    return ''.join(bases)


def insert_indel(seq: str, n_indels: int) -> str:
    bases = list(seq)
    for _ in range(n_indels):
        pos = random.randint(0, len(bases) - 1)
        if random.random() < 0.5:
            bases.insert(pos, random.choice('ACGT'))
        else:
            if len(bases) > 1:
                bases.pop(pos)
    return ''.join(bases)


def make_repeat(motif: str, copies: int, mismatch_rate: float = 0.0,
                indels_per_copy: int = 0) -> str:
    parts = []
    for _ in range(copies):
        copy = motif
        if mismatch_rate > 0:
            copy = mutate(copy, mismatch_rate)
        if indels_per_copy > 0:
            copy = insert_indel(copy, indels_per_copy)
        parts.append(copy)
    return ''.join(parts)


def write_fasta(filepath: str, name: str, seq: str):
    with open(filepath, 'w') as f:
        f.write(f">{name}\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + "\n")


def write_bed(filepath: str, records: list):
    with open(filepath, 'w') as f:
        f.write("#chrom\tstart\tend\tmotif\tcopies\ttier\n")
        for rec in records:
            f.write('\t'.join(str(x) for x in rec) + "\n")


def add_repeat(parts, truth, pos, name, motif, copies, tier,
               mismatch_rate=0.0, indels_per_copy=0):
    """Helper: append repeat and truth record, return new pos."""
    rep = make_repeat(motif, copies, mismatch_rate, indels_per_copy)
    truth.append((name, pos, pos + len(rep), motif, copies, tier))
    parts.append(rep)
    return pos + len(rep)


def add_bg(parts, pos, length, gc=0.45):
    """Helper: append background DNA, return new pos."""
    bg = random_dna(length, gc)
    parts.append(bg)
    return pos + length


# ============================================================
# Sequence 1: Tier 1 focused - short perfect STRs (~100KB)
# ============================================================
def make_seq1():
    name = "synth_tier1"
    parts = []
    truth = []
    pos = 0

    pos = add_bg(parts, pos, 8000)

    # Repeat 1: (AC)x30 = 60bp, perfect
    pos = add_repeat(parts, truth, pos, name, "AC", 30, 1)
    pos = add_bg(parts, pos, 12000)

    # Repeat 2: (ATG)x20 = 60bp, perfect
    pos = add_repeat(parts, truth, pos, name, "ATG", 20, 1)
    pos = add_bg(parts, pos, 10000)

    # Repeat 3: (AAAT)x15 = 60bp, 3% mismatch
    pos = add_repeat(parts, truth, pos, name, "AAAT", 15, 1, mismatch_rate=0.03)
    pos = add_bg(parts, pos, 9000)

    # Repeat 4: (AGCAGC)x12 = 72bp, perfect, GC-rich context
    pos = add_bg(parts, pos, 3000, gc=0.70)
    pos = add_repeat(parts, truth, pos, name, "AGCAGC", 12, 1)
    pos = add_bg(parts, pos, 3000, gc=0.70)
    pos = add_bg(parts, pos, 8000)

    # Repeat 5: (TTTC)x25 = 100bp, 5% mismatch
    pos = add_repeat(parts, truth, pos, name, "TTTC", 25, 1, mismatch_rate=0.05)
    pos = add_bg(parts, pos, 11000)

    # Repeat 6: (A)x50 = 50bp, homopolymer
    pos = add_repeat(parts, truth, pos, name, "A", 50, 1)
    pos = add_bg(parts, pos, 9000)

    # Repeat 7: (AATGG)x18 = 90bp, perfect
    pos = add_repeat(parts, truth, pos, name, "AATGG", 18, 1)
    pos = add_bg(parts, pos, 10000)

    # Repeat 8: (CT)x35 = 70bp, 4% mismatch
    pos = add_repeat(parts, truth, pos, name, "CT", 35, 1, mismatch_rate=0.04)

    # Pad to ~100KB
    remaining = 102400 - pos
    if remaining > 0:
        pos = add_bg(parts, pos, remaining)

    seq = ''.join(parts)
    write_fasta(os.path.join(SCRIPT_DIR, "synth_tier1.fa"), name, seq)
    write_bed(os.path.join(SCRIPT_DIR, "synth_tier1_truth.bed"), truth)
    print(f"seq1: {name}, len={len(seq)}, repeats={len(truth)}")


# ============================================================
# Sequence 2: Tier 2 focused - medium imperfect repeats (~100KB)
# ============================================================
def make_seq2():
    name = "synth_tier2"
    parts = []
    truth = []
    pos = 0

    pos = add_bg(parts, pos, 10000)

    # Repeat 1: 12bp motif x8, perfect
    pos = add_repeat(parts, truth, pos, name, "ACGTACGTACGT", 8, 2)
    pos = add_bg(parts, pos, 12000)

    # Repeat 2: 12bp motif x10, 5% mismatch
    pos = add_repeat(parts, truth, pos, name, "AACCGGTTAACC", 10, 2, mismatch_rate=0.05)
    pos = add_bg(parts, pos, 11000)

    # Repeat 3: 20bp random motif x6, 8% mismatch
    motif3 = random_dna(20)
    pos = add_repeat(parts, truth, pos, name, motif3, 6, 2, mismatch_rate=0.08)
    pos = add_bg(parts, pos, 13000)

    # Repeat 4: 35bp random motif x5, indels
    motif4 = random_dna(35)
    pos = add_repeat(parts, truth, pos, name, motif4, 5, 2, mismatch_rate=0.03, indels_per_copy=1)
    pos = add_bg(parts, pos, 10000)

    # Repeat 5: 50bp random motif x4, 5% divergence
    motif5 = random_dna(50)
    pos = add_repeat(parts, truth, pos, name, motif5, 4, 2, mismatch_rate=0.05)
    pos = add_bg(parts, pos, 12000)

    # Repeat 6: 18bp motif x9, perfect
    motif6 = random_dna(18)
    pos = add_repeat(parts, truth, pos, name, motif6, 9, 2)
    pos = add_bg(parts, pos, 11000)

    # Repeat 7: 25bp motif x7, 6% mismatch
    motif7 = random_dna(25)
    pos = add_repeat(parts, truth, pos, name, motif7, 7, 2, mismatch_rate=0.06)
    pos = add_bg(parts, pos, 9000)

    # Repeat 8: 40bp motif x6, 4% mismatch
    motif8 = random_dna(40)
    pos = add_repeat(parts, truth, pos, name, motif8, 6, 2, mismatch_rate=0.04)

    remaining = 102400 - pos
    if remaining > 0:
        pos = add_bg(parts, pos, remaining)

    seq = ''.join(parts)
    write_fasta(os.path.join(SCRIPT_DIR, "synth_tier2.fa"), name, seq)
    write_bed(os.path.join(SCRIPT_DIR, "synth_tier2_truth.bed"), truth)
    print(f"seq2: {name}, len={len(seq)}, repeats={len(truth)}")


# ============================================================
# Sequence 3: Tier 3 focused - long repeats (~100KB)
# ============================================================
def make_seq3():
    name = "synth_tier3"
    parts = []
    truth = []
    pos = 0

    pos = add_bg(parts, pos, 5000)

    # Repeat 1: 100bp motif x10, perfect = 1000bp
    motif1 = random_dna(100)
    pos = add_repeat(parts, truth, pos, name, motif1, 10, 3)
    pos = add_bg(parts, pos, 8000)

    # Repeat 2: 200bp motif x5, 3% mismatch = 1000bp
    motif2 = random_dna(200)
    pos = add_repeat(parts, truth, pos, name, motif2, 5, 3, mismatch_rate=0.03)
    pos = add_bg(parts, pos, 10000)

    # Repeat 3: 500bp motif x4, 5% mismatch = 2000bp
    motif3 = random_dna(500)
    pos = add_repeat(parts, truth, pos, name, motif3, 4, 3, mismatch_rate=0.05)
    pos = add_bg(parts, pos, 12000)

    # Repeat 4: 150bp motif x8, 10% divergence = 1200bp
    motif4 = random_dna(150)
    pos = add_repeat(parts, truth, pos, name, motif4, 8, 3, mismatch_rate=0.10)
    pos = add_bg(parts, pos, 10000)

    # Repeat 5: 1000bp motif x3, 8% divergence = 3000bp
    motif5 = random_dna(1000)
    pos = add_repeat(parts, truth, pos, name, motif5, 3, 3, mismatch_rate=0.08)
    pos = add_bg(parts, pos, 8000)

    # Repeat 6: 300bp motif x5, perfect = 1500bp
    motif6 = random_dna(300)
    pos = add_repeat(parts, truth, pos, name, motif6, 5, 3)
    pos = add_bg(parts, pos, 10000)

    # Repeat 7: 120bp motif x12, 4% mismatch = 1440bp
    motif7 = random_dna(120)
    pos = add_repeat(parts, truth, pos, name, motif7, 12, 3, mismatch_rate=0.04)
    pos = add_bg(parts, pos, 9000)

    # Repeat 8: 250bp motif x6, 6% mismatch = 1500bp
    motif8 = random_dna(250)
    pos = add_repeat(parts, truth, pos, name, motif8, 6, 3, mismatch_rate=0.06)

    remaining = 102400 - pos
    if remaining > 0:
        pos = add_bg(parts, pos, remaining)

    seq = ''.join(parts)
    write_fasta(os.path.join(SCRIPT_DIR, "synth_tier3.fa"), name, seq)
    write_bed(os.path.join(SCRIPT_DIR, "synth_tier3_truth.bed"), truth)
    print(f"seq3: {name}, len={len(seq)}, repeats={len(truth)}")


# ============================================================
# Sequence 4: Mixed tiers - all tiers in one sequence (~100KB)
# ============================================================
def make_seq4():
    name = "synth_mixed"
    parts = []
    truth = []
    pos = 0

    pos = add_bg(parts, pos, 6000)

    # Tier 1: (AG)x40 = 80bp
    pos = add_repeat(parts, truth, pos, name, "AG", 40, 1)
    pos = add_bg(parts, pos, 8000)

    # Tier 1: (AAAG)x18 = 72bp, 3% mismatch
    pos = add_repeat(parts, truth, pos, name, "AAAG", 18, 1, mismatch_rate=0.03)
    pos = add_bg(parts, pos, 10000)

    # Tier 2: 15bp motif x7, 5% mismatch
    motif_t2a = random_dna(15)
    pos = add_repeat(parts, truth, pos, name, motif_t2a, 7, 2, mismatch_rate=0.05)
    pos = add_bg(parts, pos, 9000)

    # Tier 2: 30bp motif x5, perfect
    motif_t2b = random_dna(30)
    pos = add_repeat(parts, truth, pos, name, motif_t2b, 5, 2)
    pos = add_bg(parts, pos, 11000)

    # Tier 1: (CAGT)x22 = 88bp, perfect
    pos = add_repeat(parts, truth, pos, name, "CAGT", 22, 1)
    pos = add_bg(parts, pos, 8000)

    # Tier 3: 200bp motif x6, 5% mismatch = 1200bp
    motif_t3a = random_dna(200)
    pos = add_repeat(parts, truth, pos, name, motif_t3a, 6, 3, mismatch_rate=0.05)
    pos = add_bg(parts, pos, 10000)

    # Tier 2: 45bp motif x5, 4% mismatch
    motif_t2c = random_dna(45)
    pos = add_repeat(parts, truth, pos, name, motif_t2c, 5, 2, mismatch_rate=0.04)
    pos = add_bg(parts, pos, 9000)

    # Tier 3: 150bp motif x4, 3% mismatch = 600bp
    motif_t3b = random_dna(150)
    pos = add_repeat(parts, truth, pos, name, motif_t3b, 4, 3, mismatch_rate=0.03)
    pos = add_bg(parts, pos, 8000)

    # Tier 1: (TG)x28 = 56bp, perfect
    pos = add_repeat(parts, truth, pos, name, "TG", 28, 1)

    remaining = 102400 - pos
    if remaining > 0:
        pos = add_bg(parts, pos, remaining)

    seq = ''.join(parts)
    write_fasta(os.path.join(SCRIPT_DIR, "synth_mixed.fa"), name, seq)
    write_bed(os.path.join(SCRIPT_DIR, "synth_mixed_truth.bed"), truth)
    print(f"seq4: {name}, len={len(seq)}, repeats={len(truth)}")


# ============================================================
# Sequence 5: Adjacent/overlapping repeats - edge cases (~100KB)
# ============================================================
def make_seq5():
    name = "synth_adjacent"
    parts = []
    truth = []
    pos = 0

    pos = add_bg(parts, pos, 8000)

    # Adjacent pair 1: (AT)x20 then 5bp gap then (AT)x15
    pos = add_repeat(parts, truth, pos, name, "AT", 20, 1)
    pos = add_bg(parts, pos, 5)  # tiny gap
    pos = add_repeat(parts, truth, pos, name, "AT", 15, 1)
    pos = add_bg(parts, pos, 12000)

    # Adjacent pair 2: (ACG)x10 immediately followed by (TGCA)x12
    pos = add_repeat(parts, truth, pos, name, "ACG", 10, 1)
    pos = add_repeat(parts, truth, pos, name, "TGCA", 12, 1)
    pos = add_bg(parts, pos, 15000)

    # Compound: Tier2 repeat
    motif_t2 = random_dna(25)
    pos = add_repeat(parts, truth, pos, name, motif_t2, 6, 2, mismatch_rate=0.04)
    pos = add_bg(parts, pos, 10000)

    # Adjacent Tier1 pair with different motifs, 8bp gap
    pos = add_repeat(parts, truth, pos, name, "AAAC", 15, 1)
    pos = add_bg(parts, pos, 8)
    pos = add_repeat(parts, truth, pos, name, "GGT", 20, 1)
    pos = add_bg(parts, pos, 12000)

    # Long repeat
    motif_long = random_dna(120)
    pos = add_repeat(parts, truth, pos, name, motif_long, 5, 3, mismatch_rate=0.03)
    pos = add_bg(parts, pos, 10000)

    # Two Tier2 repeats close together (50bp gap)
    motif_t2a = random_dna(20)
    pos = add_repeat(parts, truth, pos, name, motif_t2a, 8, 2)
    pos = add_bg(parts, pos, 50)
    motif_t2b = random_dna(18)
    pos = add_repeat(parts, truth, pos, name, motif_t2b, 7, 2, mismatch_rate=0.05)
    pos = add_bg(parts, pos, 11000)

    # Large Tier3 repeat
    motif_t3 = random_dna(300)
    pos = add_repeat(parts, truth, pos, name, motif_t3, 4, 3, mismatch_rate=0.04)

    remaining = 102400 - pos
    if remaining > 0:
        pos = add_bg(parts, pos, remaining)

    seq = ''.join(parts)
    write_fasta(os.path.join(SCRIPT_DIR, "synth_adjacent.fa"), name, seq)
    write_bed(os.path.join(SCRIPT_DIR, "synth_adjacent_truth.bed"), truth)
    print(f"seq5: {name}, len={len(seq)}, repeats={len(truth)}")


if __name__ == "__main__":
    make_seq1()
    make_seq2()
    make_seq3()
    make_seq4()
    make_seq5()
    print("Done. All synthetic fixtures generated.")
