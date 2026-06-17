import sys

def patch_file():
    with open('scripts/utils/convert_to_bed.py', 'r') as f:
        content = f.read()
    
    # 1. Replace convert_ncrf_bed with convert_ncrf
    old_ncrf_parser = '''def convert_ncrf_bed(input_file, output_file):
    """
    Parses NCRF post-processed BED format.
    chrom, start, end, name, score, strand
    For NCRF, period is usually motif length.
    """
    count = 0
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            parts = line.strip().split('\\t')
            if len(parts) >= 3:
                chrom = parts[0]
                start = parts[1]
                end = parts[2]
                # In NCRF, the user used known motifs (like TAG). 
                # We can't always deduce motif from BED if not present, so we'll use N/A.
                # If name field contains something, we could use it.
                motif = "N/A"
                period = "N/A"
                fout.write(f"{chrom}\\t{start}\\t{end}\\t{motif}\\t{period}\\tNCRF\\n")
                count += 1
    return count'''

    new_ncrf_parser = '''def convert_ncrf(input_file, output_file):
    """
    Parses raw NCRF (.ncrf) output format.
    """
    count = 0
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        chrom = ""
        start = 0
        end = 0
        sequence = ""
        
        for line in fin:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 5 and '-' in parts[3] and parts[2].endswith('bp'):
                chrom = parts[0]
                range_parts = parts[3].split('-')
                start = range_parts[0]
                end = range_parts[1]
                sequence = parts[4].replace('-', '')
            elif len(parts) >= 4 and 'score=' in parts[2]:
                motif_strand = parts[0]
                strand = motif_strand[-1] if motif_strand[-1] in ('+', '-') else '.'
                score = parts[2].replace('score=', '')
                fout.write(f"{chrom}\\t{start}\\t{end}\\t{sequence}\\t{score}\\t{strand}\\n")
                count += 1
                
    return count'''

    if old_ncrf_parser in content:
        content = content.replace(old_ncrf_parser, new_ncrf_parser)
    else:
        print("Could not find old_ncrf_parser!")
        sys.exit(1)

    # 2. Update PARSERS dict
    if "'ncrf': convert_ncrf_bed," in content:
        content = content.replace("'ncrf': convert_ncrf_bed,", "'ncrf': convert_ncrf,")
    else:
        print("Could not find ncrf in PARSERS!")
        sys.exit(1)

    # 3. Update exclusion list in main()
    old_exclude = "if f.endswith('.log') or f.endswith('.sh') or f.endswith('.settings') or f.endswith('.ncrf'):"
    new_exclude = "if f.endswith('.log') or f.endswith('.sh') or f.endswith('.settings'):"
    if old_exclude in content:
        content = content.replace(old_exclude, new_exclude)
    else:
        print("Could not find exclusion list!")
        sys.exit(1)

    with open('scripts/utils/convert_to_bed.py', 'w') as f:
        f.write(content)
    print("Patched successfully!")

if __name__ == '__main__':
    patch_file()
