"""Compare bwtandem, TRF, mreps, and ULTRA accuracy on synthetic test sequences."""
import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_ground_truth import (
    parse_truth_bed, match_repeats, canonical_motif,
    primitive_motif, periods_compatible, overlap_ratio, run_finder
)


def parse_trf_ngs(path):
    """Parse TRF -ngs output into prediction-like objects, deduplicating overlaps."""
    raw = []
    current_chrom = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('@'):
                current_chrom = line[1:]
                continue
            if not line or not current_chrom:
                continue
            parts = line.split()
            if len(parts) < 14:
                continue
            try:
                raw.append({
                    "chrom": current_chrom,
                    "start": int(parts[0]) - 1,  # TRF is 1-based
                    "end": int(parts[1]),
                    "period": int(parts[2]),
                    "copies": float(parts[3]),
                    "motif": parts[13],
                    "score": int(parts[7]),
                })
            except (ValueError, IndexError):
                continue
    # Deduplicate: for overlapping records, keep highest-scoring one
    raw.sort(key=lambda r: r["score"], reverse=True)
    kept = []
    for r in raw:
        dominated = False
        for k in kept:
            if r["chrom"] == k["chrom"]:
                ov_start = max(r["start"], k["start"])
                ov_end = min(r["end"], k["end"])
                if ov_end > ov_start:
                    ov_len = ov_end - ov_start
                    min_len = min(r["end"] - r["start"], k["end"] - k["start"])
                    if ov_len / min_len > 0.5:
                        dominated = True
                        break
        if not dominated:
            kept.append(r)
    return kept


def parse_mreps_output(path):
    """Parse mreps output into prediction-like objects."""
    records = []
    chrom = None
    with open(path) as f:
        for line in f:
            m_proc = re.match(r"Processing sequence '(\S+)'", line)
            if m_proc:
                chrom = m_proc.group(1)
                continue
            m = re.match(r'^\s+(\d+)\s+->\s+(\d+)\s+:\s+(\d+)\s+<(\d+)>\s+\[[\d.]+\]\s+([\d.]+)', line)
            if m and chrom:
                start = int(m.group(1)) - 1  # mreps is 1-based
                end = int(m.group(2))
                size = int(m.group(3))
                period = int(m.group(4))
                err_rate = float(m.group(5))
                records.append({
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "period": period,
                    "copies": size / period if period > 0 else 0,
                    "motif": "N" * period,  # mreps doesn't always give clean motif
                })
    return records


def parse_ultra_bed(path):
    """Parse ULTRA BED/TSV output."""
    records = []
    with open(path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue
            try:
                records.append({
                    "chrom": parts[0],
                    "start": int(parts[1]),
                    "end": int(parts[2]),
                    "period": int(parts[3]),
                    "copies": (int(parts[2]) - int(parts[1])) / int(parts[3]) if int(parts[3]) > 0 else 0,
                    "motif": parts[5],
                })
            except (ValueError, IndexError):
                continue
    return records


class FakePred:
    """Wrapper to make tool records look like bwtandem predictions."""
    def __init__(self, r):
        self.start = r["start"]
        self.end = r["end"]
        self.motif = r["motif"]
        self.consensus_motif = r["motif"]
        self.copies = r["copies"]
        self.tier = 0


def evaluate_tool(truth, records):
    """Evaluate tool predictions against ground truth."""
    preds = [FakePred(r) for r in records]
    tp, fp, fn, matched, extras = match_repeats(truth, preds)
    n_truth = len(truth)
    sens = tp / n_truth if n_truth else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * sens * prec / (sens + prec) if (sens + prec) else 0
    return tp, fp, fn, sens, prec, f1


def main():
    fixtures = [
        ("synth_tier1", "tests/fixtures/synth_tier1_truth.bed", {"tier1"}),
        ("synth_tier2", "tests/fixtures/synth_tier2_truth.bed", {"tier1", "tier2"}),
        ("synth_tier3", "tests/fixtures/synth_tier3_truth.bed", {"tier1", "tier2", "tier3"}),
        ("synth_mixed", "tests/fixtures/synth_mixed_truth.bed", {"tier1", "tier2", "tier3"}),
        ("synth_adjacent", "tests/fixtures/synth_adjacent_truth.bed", {"tier1", "tier2", "tier3"}),
    ]

    tools = ["bwtandem", "TRF", "mreps", "ULTRA"]
    header = f"{'Fixture':<16} {'Truth':>5}"
    for tool in tools:
        header += f" | {'TP':>3} {'FP':>4} {'FN':>3} {'Sens':>6} {'Prec':>6}"
    print(header)
    print(f"{'':16} {'':>5}", end="")
    for tool in tools:
        print(f" | {tool:^28}", end="")
    print()
    print("-" * (22 + 31 * len(tools)))

    totals = {t: {"tp": 0, "fp": 0, "fn": 0, "truth": 0} for t in tools}

    for name, truth_bed, bwt_tiers in fixtures:
        truth = parse_truth_bed(truth_bed)
        fa_path = truth_bed.replace('_truth.bed', '.fa')
        n_truth = len(truth)

        results = {}

        # bwtandem
        bwt_preds = run_finder(fa_path, bwt_tiers)
        tp, fp, fn, sens, prec, f1 = evaluate_tool(truth, [
            {"chrom": p.start, "start": p.start, "end": p.end,
             "motif": p.motif, "copies": p.copies, "period": len(p.motif)}
            for p in bwt_preds
        ])
        # Re-do with actual match_repeats for bwtandem (it uses the real pred objects)
        tp, fp, fn, _, _ = match_repeats(truth, bwt_preds)
        sens = tp / n_truth if n_truth else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        results["bwtandem"] = (tp, fp, fn, sens, prec)

        # TRF
        trf_path = f"results/trf_{name}_ngs.txt"
        if os.path.exists(trf_path):
            trf_records = parse_trf_ngs(trf_path)
            tp, fp, fn, sens, prec, f1 = evaluate_tool(truth, trf_records)
            results["TRF"] = (tp, fp, fn, sens, prec)
        else:
            results["TRF"] = (0, 0, n_truth, 0, 0)

        # mreps
        mreps_path = f"results/mreps_{name}.txt"
        if os.path.exists(mreps_path):
            mreps_records = parse_mreps_output(mreps_path)
            tp, fp, fn, sens, prec, f1 = evaluate_tool(truth, mreps_records)
            results["mreps"] = (tp, fp, fn, sens, prec)
        else:
            results["mreps"] = (0, 0, n_truth, 0, 0)

        # ULTRA
        ultra_path = f"results/ultra_{name}.bed"
        if os.path.exists(ultra_path):
            ultra_records = parse_ultra_bed(ultra_path)
            tp, fp, fn, sens, prec, f1 = evaluate_tool(truth, ultra_records)
            results["ULTRA"] = (tp, fp, fn, sens, prec)
        else:
            results["ULTRA"] = (0, 0, n_truth, 0, 0)

        # Print row
        line = f"{name:<16} {n_truth:>5}"
        for tool in tools:
            tp, fp, fn, sens, prec = results[tool]
            line += f" | {tp:>3} {fp:>4} {fn:>3} {sens:>5.1%} {prec:>5.1%}"
            totals[tool]["tp"] += tp
            totals[tool]["fp"] += fp
            totals[tool]["fn"] += fn
            totals[tool]["truth"] += n_truth
        print(line)

    # Print totals
    print("-" * (22 + 31 * len(tools)))
    line = f"{'TOTAL':<16} {totals['bwtandem']['truth']:>5}"
    for tool in tools:
        t = totals[tool]
        s = t["tp"] / t["truth"] if t["truth"] else 0
        p = t["tp"] / (t["tp"] + t["fp"]) if (t["tp"] + t["fp"]) else 0
        f1 = 2 * s * p / (s + p) if (s + p) else 0
        line += f" | {t['tp']:>3} {t['fp']:>4} {t['fn']:>3} {s:>5.1%} {p:>5.1%}"
    print(line)

    # Summary table for README
    print("\n\n=== README Table ===")
    print("| Tool | Sensitivity | Precision | F1 |")
    print("|------|-------------|-----------|-----|")
    for tool in tools:
        t = totals[tool]
        s = t["tp"] / t["truth"] if t["truth"] else 0
        p = t["tp"] / (t["tp"] + t["fp"]) if (t["tp"] + t["fp"]) else 0
        f1 = 2 * s * p / (s + p) if (s + p) else 0
        print(f"| {tool} | {s:.1%} | {p:.1%} | {f1:.1%} |")


if __name__ == "__main__":
    main()
