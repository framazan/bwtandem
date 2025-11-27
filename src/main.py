import argparse
import sys
import time
import os
from typing import List, Iterator, Tuple
from .finder import TandemRepeatFinder
from .models import TandemRepeat

def parse_fasta(file_path: str) -> Iterator[Tuple[str, str]]:
    """Simple FASTA parser to avoid Biopython dependency."""
    name = None
    seq_parts = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if name:
                    yield name, "".join(seq_parts)
                name = line[1:].split()[0]  # Take first word as ID
                seq_parts = []
            else:
                seq_parts.append(line)
        if name:
            yield name, "".join(seq_parts)

def main():
    parser = argparse.ArgumentParser(description="BWT-based Tandem Repeat Finder")
    parser.add_argument("fasta_file", help="Input FASTA file")
    parser.add_argument("--min-period", type=int, default=1, help="Minimum period size (default: 1)")
    parser.add_argument("--max-period", type=int, default=2000, help="Maximum period size (default: 2000)")
    parser.add_argument("--min-array-bp", type=int, default=None,
                        help="Minimum repeat array length in bp (default: no minimum)")
    parser.add_argument("--max-array-bp", type=int, default=None,
                        help="Maximum repeat array length in bp (default: no maximum)")
    parser.add_argument("--tiers", type=str, default="tier1,tier2,tier3",
                        help="Comma-separated list of tiers to run (tier1,tier2,tier3) or 'all'")
    parser.add_argument("--output", "-o", help="Output file prefix (default: input filename)")
    parser.add_argument("--format", choices=["bed", "vcf", "trf", "strfinder"], default="bed", help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show progress")
    parser.add_argument("--profile", action="store_true", help="Profile execution with cProfile and print top hotspots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.fasta_file):
        print(f"Error: File {args.fasta_file} not found")
        sys.exit(1)
        
    output_prefix = args.output if args.output else os.path.splitext(args.fasta_file)[0]
    out_file = f"{output_prefix}.{args.format}" # Default output filename
    
    print(f"Processing {args.fasta_file}...")
    start_total = time.time()
    
    all_repeats: List[TandemRepeat] = []
    
    tiers_arg = args.tiers.strip()
    if tiers_arg.lower() == "all":
        enabled_tiers = {"tier1", "tier2", "tier3"}
    else:
        enabled_tiers = {t.strip().lower() for t in tiers_arg.split(',') if t.strip()}

    # Optional profiler
    profiler = None
    if args.profile:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()

    for chrom, seq in parse_fasta(args.fasta_file):
        seq = seq.upper()
        
        if args.verbose:
            print(f"Processing sequence: {chrom} ({len(seq)} bp)")
            
        finder = TandemRepeatFinder(
            seq, 
            chromosome=chrom,
            min_period=args.min_period,
            max_period=args.max_period,
            show_progress=args.verbose,
            enabled_tiers=enabled_tiers,
            min_array_bp=args.min_array_bp,
            max_array_bp=args.max_array_bp
        )
        
        repeats = finder.find_all()
        all_repeats.extend(repeats)
        
        finder.cleanup()
        
    # Stop profiler and report
    if profiler is not None:
        profiler.disable()
    
    print(f"Total repeats found: {len(all_repeats)}")
    print(f"Total time: {time.time() - start_total:.2f}s")
    
    if profiler is not None:
        import pstats
        profile_path = f"{output_prefix}.tier2_profile.prof"
        profiler.dump_stats(profile_path)
        print(f"Profile written to {profile_path}")
        print("Top 20 cumulative time hotspots:")
        stats = pstats.Stats(profiler)
        stats.strip_dirs().sort_stats("cumulative").print_stats(20)
    
    # Write output
    if args.format == "bed":
        out_file = f"{output_prefix}.bed"
        with open(out_file, "w") as f:
            for r in all_repeats:
                f.write(r.to_bed() + "\n")
    elif args.format == "vcf":
        out_file = f"{output_prefix}.vcf"
        with open(out_file, "w") as f:
            f.write("##fileformat=VCFv4.2\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
            for r in all_repeats:
                # Construct VCF line
                ref = r.actual_sequence if r.actual_sequence else "N"
                alt = "<STR>"
                info = r.to_vcf_info()
                f.write(f"{r.chrom}\t{r.start+1}\t.\t{ref}\t{alt}\t.\t.\t{info}\n")
    elif args.format == "trf":
        out_file = f"{output_prefix}.dat"
        with open(out_file, "w") as f:
            for r in all_repeats:
                f.write(r.to_trf_dat() + "\n")
    elif args.format == "strfinder":
        out_file = f"{output_prefix}.csv"
        with open(out_file, "w") as f:
            f.write("STR_marker\tSTR_position\tSTR_motif\tSTR_genotype_structure\tSTR_genotype\tSTR_core_seq\tAllele_coverage\tAlleles_ratio\tReads_Distribution\tSTR_depth\tFull_seq\tVariations\n")
            for r in all_repeats:
                f.write(r.to_strfinder() + "\n")
                
    print(f"Results written to {out_file}")

if __name__ == "__main__":
    main()
