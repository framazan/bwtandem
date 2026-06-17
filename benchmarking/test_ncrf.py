import sys

def convert_ncrf(input_file):
    count = 0
    with open(input_file, 'r') as fin:
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
                motif_name = motif_strand[:-1] if strand != '.' else motif_strand
                score = parts[2].replace('score=', '')
                period = str(len(motif_name))
                print(f"{chrom}\t{start}\t{end}\t{sequence[:20]}...\t{score}\t{strand}\t{motif_name}")
                count += 1
                if count >= 5:
                    break

if __name__ == '__main__':
    convert_ncrf(sys.argv[1])
