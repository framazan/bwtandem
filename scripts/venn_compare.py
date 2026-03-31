"""Compare bwtandem, TRF, mreps, and ULTRA results using Venn diagrams.

Compares tool outputs on Chr4 by overlapping genomic regions.
"""
import os
import re
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3


def parse_bed(path):
    """Parse BED-like file: chrom, start, end."""
    regions = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            try:
                regions.append((parts[0], int(parts[1]), int(parts[2])))
            except (ValueError, IndexError):
                continue
    return regions


def parse_mreps(path):
    """Parse mreps output to extract regions.

    Format: '    1386  ->      1407 :  ...'
    """
    regions = []
    chrom = "Chr4"
    with open(path) as f:
        for line in f:
            # mreps format: "   1386  ->      1407 :  ..."
            m = re.match(r'^\s+(\d+)\s+->\s+(\d+)\s+:', line)
            if m:
                regions.append((chrom, int(m.group(1)), int(m.group(2))))
    return regions


def parse_ultra(path):
    """Parse ULTRA TSV output."""
    regions = []
    with open(path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                try:
                    regions.append((parts[0], int(parts[1]), int(parts[2])))
                except (ValueError, IndexError):
                    continue
    return regions


def regions_overlap(r1, r2, min_overlap_ratio=0.5):
    """Check if two regions overlap by at least min_overlap_ratio."""
    if r1[0] != r2[0]:
        return False
    overlap_start = max(r1[1], r2[1])
    overlap_end = min(r1[2], r2[2])
    if overlap_start >= overlap_end:
        return False
    overlap_len = overlap_end - overlap_start
    min_len = min(r1[2] - r1[1], r2[2] - r2[1])
    return overlap_len / min_len >= min_overlap_ratio


def find_shared(regions_a, regions_b, min_overlap=0.3):
    """Find number of regions in A that overlap with any region in B."""
    shared = 0
    for ra in regions_a:
        for rb in regions_b:
            if regions_overlap(ra, rb, min_overlap):
                shared += 1
                break
    return shared


def bin_regions(regions, bin_size=500):
    """Bin regions into genomic windows for set-based comparison."""
    bins = set()
    for chrom, start, end in regions:
        for pos in range(start // bin_size, end // bin_size + 1):
            bins.add((chrom, pos))
    return bins


def main():
    results_dir = "results"

    # Parse all tool results
    tools = {}

    bwt_path = os.path.join(results_dir, "bwt_Chr4.bed")
    if os.path.exists(bwt_path):
        tools["bwtandem"] = parse_bed(bwt_path)
        print(f"bwtandem: {len(tools['bwtandem'])} repeats")

    trf_path = os.path.join(results_dir, "trf_Chr4.bed")
    if os.path.exists(trf_path):
        tools["TRF"] = parse_bed(trf_path)
        print(f"TRF: {len(tools['TRF'])} repeats")

    mreps_path = os.path.join(results_dir, "mreps_Chr4.txt")
    if os.path.exists(mreps_path):
        tools["mreps"] = parse_mreps(mreps_path)
        print(f"mreps: {len(tools['mreps'])} repeats")

    ultra_path = os.path.join(results_dir, "ultra_Chr4.tsv")
    if os.path.exists(ultra_path):
        tools["ULTRA"] = parse_ultra(ultra_path)
        print(f"ULTRA: {len(tools['ULTRA'])} repeats")

    if len(tools) < 2:
        print("Need at least 2 tool results for comparison")
        sys.exit(1)

    # Use genomic bins (500bp windows) for set-based Venn comparison
    binned = {name: bin_regions(regions) for name, regions in tools.items()}

    # Print overlap matrix
    tool_names = list(tools.keys())
    print(f"\n{'Overlap Matrix (shared regions)':=^60}")
    print(f"{'':>12}", end="")
    for name in tool_names:
        print(f"{name:>12}", end="")
    print()
    for n1 in tool_names:
        print(f"{n1:>12}", end="")
        for n2 in tool_names:
            shared = find_shared(tools[n1], tools[n2], min_overlap=0.3)
            print(f"{shared:>12}", end="")
        print()

    # Create Venn diagrams
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- 2-way Venn: bwtandem vs TRF ---
    if "bwtandem" in binned and "TRF" in binned:
        ax = axes[0]
        bwt_set = binned["bwtandem"]
        trf_set = binned["TRF"]
        v2 = venn2([bwt_set, trf_set], set_labels=("bwtandem", "TRF"), ax=ax)
        ax.set_title("bwtandem vs TRF\n(Chr4, 500bp genomic bins)", fontsize=13)

    # --- 3-way Venn: bwtandem vs TRF vs ULTRA/mreps ---
    third_tool = None
    for t in ["ULTRA", "mreps"]:
        if t in binned:
            third_tool = t
            break

    if "bwtandem" in binned and "TRF" in binned and third_tool:
        ax = axes[1]
        v3 = venn3(
            [binned["bwtandem"], binned["TRF"], binned[third_tool]],
            set_labels=("bwtandem", "TRF", third_tool),
            ax=ax
        )
        ax.set_title(f"bwtandem vs TRF vs {third_tool}\n(Chr4, 500bp genomic bins)", fontsize=13)

    plt.tight_layout()
    out_path = os.path.join(results_dir, "venn_comparison_chr4.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nVenn diagram saved to: {out_path}")

    # Also create all-4-tool pairwise overlap summary
    if len(tools) >= 3:
        print(f"\n{'Pairwise Overlap Summary':=^60}")
        for i, n1 in enumerate(tool_names):
            for n2 in tool_names[i+1:]:
                shared_12 = find_shared(tools[n1], tools[n2], min_overlap=0.3)
                shared_21 = find_shared(tools[n2], tools[n1], min_overlap=0.3)
                print(f"  {n1} ∩ {n2}: {shared_12} of {len(tools[n1])} ({100*shared_12/max(len(tools[n1]),1):.1f}%) | "
                      f"{shared_21} of {len(tools[n2])} ({100*shared_21/max(len(tools[n2]),1):.1f}%)")


if __name__ == "__main__":
    main()
