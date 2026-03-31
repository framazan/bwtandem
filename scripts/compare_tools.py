"""Compare bwtandem vs TRF accuracy on synthetic test sequences."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_ground_truth import (
    parse_truth_bed, match_repeats, canonical_motif,
    primitive_motif, periods_compatible, overlap_ratio, run_finder
)


def parse_trf_bed(path):
    """Parse TRF BED output into truth-like dicts."""
    records = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            records.append({
                "chrom": parts[0],
                "start": int(parts[1]),
                "end": int(parts[2]),
                "motif": parts[3],
                "copies": float(parts[4]),
                "tier": 0,  # TRF doesn't have tiers
            })
    return records


def compare_on_fixture(name, truth_bed, bwt_tiers):
    """Compare bwtandem and TRF on a single fixture."""
    truth = parse_truth_bed(truth_bed)
    fa_path = truth_bed.replace('_truth.bed', '.fa')

    # bwtandem results
    bwt_preds = run_finder(fa_path, bwt_tiers)
    bwt_tp, bwt_fp, bwt_fn, _, _ = match_repeats(truth, bwt_preds)

    # TRF results
    trf_bed = os.path.join('results', f'trf_{name}.bed')
    if not os.path.exists(trf_bed):
        return None
    trf_records = parse_trf_bed(trf_bed)

    # Convert TRF records to prediction-like objects for matching
    class FakePred:
        def __init__(self, r):
            self.start = r["start"]
            self.end = r["end"]
            self.motif = r["motif"]
            self.consensus_motif = r["motif"]
            self.copies = r["copies"]
            self.tier = r["tier"]

    trf_preds = [FakePred(r) for r in trf_records]
    trf_tp, trf_fp, trf_fn, _, _ = match_repeats(truth, trf_preds)

    n_truth = len(truth)
    bwt_sens = bwt_tp / n_truth if n_truth else 0
    bwt_prec = bwt_tp / (bwt_tp + bwt_fp) if (bwt_tp + bwt_fp) else 0
    trf_sens = trf_tp / n_truth if n_truth else 0
    trf_prec = trf_tp / (trf_tp + trf_fp) if (trf_tp + trf_fp) else 0

    return {
        "name": name,
        "truth": n_truth,
        "bwt_tp": bwt_tp, "bwt_fp": bwt_fp, "bwt_fn": bwt_fn,
        "bwt_sens": bwt_sens, "bwt_prec": bwt_prec,
        "trf_tp": trf_tp, "trf_fp": trf_fp, "trf_fn": trf_fn,
        "trf_sens": trf_sens, "trf_prec": trf_prec,
    }


def main():
    fixtures = [
        ("synth_tier1", "tests/fixtures/synth_tier1_truth.bed", {"tier1"}),
        ("synth_tier2", "tests/fixtures/synth_tier2_truth.bed", {"tier1", "tier2"}),
        ("synth_tier3", "tests/fixtures/synth_tier3_truth.bed", {"tier1", "tier2", "tier3"}),
        ("synth_mixed", "tests/fixtures/synth_mixed_truth.bed", {"tier1", "tier2", "tier3"}),
        ("synth_adjacent", "tests/fixtures/synth_adjacent_truth.bed", {"tier1", "tier2", "tier3"}),
    ]

    print(f"{'Fixture':<16} {'Truth':>5} | {'bwtandem':^25} | {'TRF':^25}")
    print(f"{'':16} {'':>5} | {'TP':>4} {'FP':>4} {'FN':>4} {'Sens':>6} {'Prec':>6} | {'TP':>4} {'FP':>4} {'FN':>4} {'Sens':>6} {'Prec':>6}")
    print("-" * 85)

    totals = {"bwt_tp": 0, "bwt_fp": 0, "bwt_fn": 0,
              "trf_tp": 0, "trf_fp": 0, "trf_fn": 0, "truth": 0}

    for name, truth_bed, tiers in fixtures:
        result = compare_on_fixture(name, truth_bed, tiers)
        if result is None:
            print(f"{name:<16} — TRF results not found")
            continue

        r = result
        print(f"{r['name']:<16} {r['truth']:>5} | "
              f"{r['bwt_tp']:>4} {r['bwt_fp']:>4} {r['bwt_fn']:>4} "
              f"{r['bwt_sens']:>5.1%} {r['bwt_prec']:>5.1%} | "
              f"{r['trf_tp']:>4} {r['trf_fp']:>4} {r['trf_fn']:>4} "
              f"{r['trf_sens']:>5.1%} {r['trf_prec']:>5.1%}")

        for k in totals:
            totals[k] += r[k]

    print("-" * 85)
    bwt_s = totals["bwt_tp"] / totals["truth"] if totals["truth"] else 0
    bwt_p = totals["bwt_tp"] / (totals["bwt_tp"] + totals["bwt_fp"]) if (totals["bwt_tp"] + totals["bwt_fp"]) else 0
    trf_s = totals["trf_tp"] / totals["truth"] if totals["truth"] else 0
    trf_p = totals["trf_tp"] / (totals["trf_tp"] + totals["trf_fp"]) if (totals["trf_tp"] + totals["trf_fp"]) else 0
    print(f"{'TOTAL':<16} {totals['truth']:>5} | "
          f"{totals['bwt_tp']:>4} {totals['bwt_fp']:>4} {totals['bwt_fn']:>4} "
          f"{bwt_s:>5.1%} {bwt_p:>5.1%} | "
          f"{totals['trf_tp']:>4} {totals['trf_fp']:>4} {totals['trf_fn']:>4} "
          f"{trf_s:>5.1%} {trf_p:>5.1%}")


if __name__ == "__main__":
    main()
