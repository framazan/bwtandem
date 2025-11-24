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
    parser.add_argument("--output", "-o", help="Output file prefix (default: input filename)")
    parser.add_argument("--format", choices=["bed", "vcf", "trf", "strfinder"], default="bed", help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show progress")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.fasta_file):
        print(f"Error: File {args.fasta_file} not found")
        sys.exit(1)
        
    output_prefix = args.output if args.output else os.path.splitext(args.fasta_file)[0]
    out_file = f"{output_prefix}.{args.format}" # Default output filename
    
    print(f"Processing {args.fasta_file}...")
    start_total = time.time()
    
    all_repeats: List[TandemRepeat] = []
    
    for chrom, seq in parse_fasta(args.fasta_file):
        seq = seq.upper()
        
        if args.verbose:
            print(f"Processing sequence: {chrom} ({len(seq)} bp)")
            
        finder = TandemRepeatFinder(
            seq, 
            chromosome=chrom,
            min_period=args.min_period,
            max_period=args.max_period,
            show_progress=args.verbose
        )
        
        repeats = finder.find_all()
        all_repeats.extend(repeats)
        
        finder.cleanup()
        
    print(f"Total repeats found: {len(all_repeats)}")
    print(f"Total time: {time.time() - start_total:.2f}s")
    
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
