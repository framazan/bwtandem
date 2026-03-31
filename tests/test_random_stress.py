#!/usr/bin/env python3
"""Stress test: generate 30 random synthetic sequences with varied repeats,
run TandemRepeatFinder on each, and report aggregate sensitivity/precision.

This file is temporary — run once to validate robustness, then delete.
"""
import os
import sys
import random
import tempfile
import shutil
import numpy as np

# Reuse helpers from generate_synthetic
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures"))
from tests.fixtures.generate_synthetic import random_dna, make_repeat, write_fasta, write_bed

# Reuse test helpers
from tests.test_ground_truth import (
    parse_truth_bed, parse_fasta_simple, run_finder,
    match_repeats, compute_metrics, canonical_motif, primitive_motif,
    report_results
)

NUM_SEQUENCES = 30
SEQ_LENGTH = 20000  # 20KB per sequence (fast stress test)

# Motif pools for each tier
TIER1_MOTIFS = [
    ("AC", 25, 0.0), ("ATG", 18, 0.0), ("AAAT", 12, 0.03),
    ("TGCA", 20, 0.0), ("CT", 30, 0.04), ("A", 40, 0.0),
    ("AATGG", 14, 0.0), ("GGC", 15, 0.02), ("CAGT", 18, 0.0),
    ("TG", 22, 0.05), ("AAAC", 16, 0.0), ("TAG", 20, 0.03),
    ("CCTA", 10, 0.0), ("AGGT", 15, 0.02), ("GAT", 25, 0.0),
]

TIER2_PERIODS = [10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50]
TIER2_COPIES = [4, 5, 6, 7, 8, 9, 10]
TIER2_MM = [0.0, 0.02, 0.03, 0.05, 0.06, 0.08]

TIER3_PERIODS = [100, 120, 150, 200, 250, 300, 500]
TIER3_COPIES = [3, 4, 5, 6, 8]
TIER3_MM = [0.0, 0.03, 0.05, 0.08, 0.10]


def generate_random_sequence(seed_val, tmpdir, seq_idx):
    """Generate one random sequence with 2-5 repeats from mixed tiers."""
    rng = random.Random(seed_val)
    name = f"stress_{seq_idx:03d}"
    parts = []
    truth = []
    pos = 0

    # Random number of repeats (2-5)
    n_repeats = rng.randint(2, 5)

    # Distribute background between repeats
    bg_total = SEQ_LENGTH - 500  # rough estimate, pad at end
    bg_per = bg_total // (n_repeats + 1)

    for i in range(n_repeats):
        # Add background
        bg_len = max(1000, bg_per + rng.randint(-2000, 2000))
        bg = random_dna(bg_len, gc=rng.uniform(0.35, 0.55))
        parts.append(bg)
        pos += bg_len

        # Choose tier randomly
        tier = rng.choice([1, 1, 1, 2, 2, 3])  # weighted toward tier 1

        if tier == 1:
            motif, copies, mm = rng.choice(TIER1_MOTIFS)
            # Vary copies a bit
            copies = max(3, copies + rng.randint(-3, 3))
            rep = make_repeat(motif, copies, mm)
            truth.append((name, pos, pos + len(rep), motif, copies, 1))
            parts.append(rep)
            pos += len(rep)

        elif tier == 2:
            period = rng.choice(TIER2_PERIODS)
            motif = random_dna(period, gc=rng.uniform(0.35, 0.55))
            # Ensure primitive period
            while len(set(motif[j::period] for j in range(min(period, 4)))) < 2:
                motif = random_dna(period)
            copies = rng.choice(TIER2_COPIES)
            mm = rng.choice(TIER2_MM)
            rep = make_repeat(motif, copies, mm)
            truth.append((name, pos, pos + len(rep), motif, copies, 2))
            parts.append(rep)
            pos += len(rep)

        else:  # tier 3
            period = rng.choice(TIER3_PERIODS)
            motif = random_dna(period, gc=rng.uniform(0.35, 0.55))
            copies = rng.choice(TIER3_COPIES)
            mm = rng.choice(TIER3_MM)
            rep = make_repeat(motif, copies, mm)
            truth.append((name, pos, pos + len(rep), motif, copies, 3))
            parts.append(rep)
            pos += len(rep)

    # Pad to target length
    remaining = max(0, SEQ_LENGTH - pos)
    if remaining > 0:
        parts.append(random_dna(remaining))
        pos += remaining

    seq = ''.join(parts)

    fa_path = os.path.join(tmpdir, f"{name}.fa")
    bed_path = os.path.join(tmpdir, f"{name}_truth.bed")
    write_fasta(fa_path, name, seq)
    write_bed(bed_path, truth)

    return fa_path, bed_path, len(truth)


def main():
    tmpdir = tempfile.mkdtemp(prefix="bwtandem_stress_")
    print(f"Temp dir: {tmpdir}")
    print(f"Generating {NUM_SEQUENCES} random sequences...")

    # Aggregate counters
    total_tp = {1: 0, 2: 0, 3: 0, "all": 0}
    total_fp = {1: 0, 2: 0, 3: 0, "all": 0}
    total_fn = {1: 0, 2: 0, 3: 0, "all": 0}
    total_truth = {1: 0, 2: 0, 3: 0, "all": 0}

    for i in range(NUM_SEQUENCES):
        seed_val = 1000 + i
        fa_path, bed_path, n_truth = generate_random_sequence(seed_val, tmpdir, i)
        truth = parse_truth_bed(bed_path)

        # Always enable all tiers (matches real-world usage where
        # cross-tier detection with period-compatible matching is valid)
        enabled = {"tier1", "tier2", "tier3"}

        try:
            preds = run_finder(fa_path, enabled)
        except Exception as e:
            print(f"  [{i:03d}] ERROR: {e}")
            continue

        tp, fp, fn, missed, extra = match_repeats(truth, preds)
        metrics = compute_metrics(tp, fp, fn)

        total_tp["all"] += tp
        total_fp["all"] += fp
        total_fn["all"] += fn
        total_truth["all"] += len(truth)

        # Per-tier breakdown
        for tier_num in [1, 2, 3]:
            tier_truth = [t for t in truth if t["tier"] == tier_num]
            if tier_truth:
                t_tp, t_fp, t_fn, _, _ = match_repeats(tier_truth, preds)
                total_tp[tier_num] += t_tp
                total_fn[tier_num] += t_fn
                total_truth[tier_num] += len(tier_truth)

        status = "OK" if metrics["sensitivity"] >= 0.7 else "LOW"
        print(f"  [{i:03d}] TP={tp} FP={fp} FN={fn} "
              f"sens={metrics['sensitivity']:.0%} prec={metrics['precision']:.0%} "
              f"[{status}]"
              + (f" missed: {[(m['motif'][:10], m['start']) for m in missed]}" if missed else ""))

    # Summary
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    for label, key in [("Overall", "all"), ("Tier 1", 1), ("Tier 2", 2), ("Tier 3", 3)]:
        tp = total_tp[key]
        fn = total_fn[key]
        truth_n = total_truth[key]
        if key == "all":
            fp = total_fp[key]
            m = compute_metrics(tp, fp, fn)
            print(f"  {label:10s}: TP={tp:3d} FP={fp:3d} FN={fn:3d} "
                  f"Sensitivity={m['sensitivity']:.1%} "
                  f"Precision={m['precision']:.1%} "
                  f"F1={m['f1']:.1%} "
                  f"(total truth={truth_n})")
        else:
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            print(f"  {label:10s}: TP={tp:3d} FN={fn:3d} "
                  f"Sensitivity={sens:.1%} "
                  f"(total truth={truth_n})")

    # Cleanup
    shutil.rmtree(tmpdir)
    print(f"\nCleaned up {tmpdir}")

    # Final pass/fail
    overall = compute_metrics(total_tp["all"], total_fp["all"], total_fn["all"])
    if overall["sensitivity"] >= 0.80 and overall["precision"] >= 0.50:
        print("\n*** STRESS TEST PASSED ***")
        return 0
    else:
        print(f"\n*** STRESS TEST FAILED *** "
              f"(sens={overall['sensitivity']:.1%}, prec={overall['precision']:.1%})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
