import sys

def convert_trf_ngs_to_bed(dat_file, bed_file) -> int:
    repeat_count = 0
    with open(dat_file, 'r') as f_in, open(bed_file, 'w') as f_out:
        current_chrom = "unknown"
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
                
            # TRF -ngs format prepends sequence headers with @
            if line.startswith('@'):
                # Extract the chromosome name (e.g. '@Chr4' -> 'Chr4')
                current_chrom = line[1:].strip().split()[0]
                continue
                
            # Skip any malformed or parameter lines
            parts = line.split()
            if len(parts) < 15 or not parts[0].isdigit():
                continue
                
            try:
                # Based on TRF DAT format columns:
                # 0: Start, 1: End, 2: Period, 3: Copies, 4: Consensus Size,
                # 5: % Matches, 6: % Indels, 7: Score, 8-11: A, C, G, T%,
                # 12: Entropy, 13: Consensus Motif, 14: Actual Sequence
                start = int(parts[0]) - 1 # TRF is 1-indexed, BED is 0-indexed
                end = int(parts[1])
                period = parts[2]
                copies = float(parts[3])
                motif = parts[13]
                
                # We can calculate an approximate mismatch rate string to match your pipeline
                percent_matches = float(parts[5])
                mismatch_rate = (100.0 - percent_matches) / 100.0
                
                # Format: Chrom Start End Motif Copies Tier Mismatch-Rate Strand
                bed_line = f"{current_chrom}\t{start}\t{end}\t{motif}\t{copies:.1f}\tTRF\t{mismatch_rate:.3f}\t+\n"
                f_out.write(bed_line)
                repeat_count += 1
                
            except ValueError:
                # If a line fails to parse, quietly skip it (it might be a header or parameter comment)
                continue
                
    return repeat_count

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python trf_to_bed.py <input.dat> <output.bed>")
        sys.exit(1)
        
    convert_trf_ngs_to_bed(sys.argv[1], sys.argv[2])
    print(f"Successfully converted {sys.argv[1]} to {sys.argv[2]}")
