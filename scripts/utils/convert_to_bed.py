#!/usr/bin/env python3
import argparse
import sys
import os
import glob
import csv

def clean_chrom(chrom):
    # Mapping for RefSeq GRCh38 accessions
    refseq_map = {
        'NC_000001.11': 'chr1',
        'NC_000002.12': 'chr2',
        'NC_000003.12': 'chr3',
        'NC_000004.12': 'chr4',
        'NC_000005.10': 'chr5',
        'NC_000006.12': 'chr6',
        'NC_000007.14': 'chr7',
        'NC_000008.11': 'chr8',
        'NC_000009.12': 'chr9',
        'NC_000010.11': 'chr10',
        'NC_000011.10': 'chr11',
        'NC_000012.12': 'chr12',
        'NC_000013.11': 'chr13',
        'NC_000014.9': 'chr14',
        'NC_000015.10': 'chr15',
        'NC_000016.10': 'chr16',
        'NC_000017.11': 'chr17',
        'NC_000018.10': 'chr18',
        'NC_000019.10': 'chr19',
        'NC_000020.11': 'chr20',
        'NC_000021.9': 'chr21',
        'NC_000022.11': 'chr22',
        'NC_000023.11': 'chrX',
        'NC_000024.10': 'chrY',
    }
    # Mapping for GenBank GRCh38 accessions
    genbank_map = {
        'CM000663.2': 'chr1',
        'CM000664.2': 'chr2',
        'CM000665.2': 'chr3',
        'CM000666.2': 'chr4',
        'CM000667.2': 'chr5',
        'CM000668.2': 'chr6',
        'CM000669.2': 'chr7',
        'CM000670.2': 'chr8',
        'CM000671.2': 'chr9',
        'CM000672.2': 'chr10',
        'CM000673.2': 'chr11',
        'CM000674.2': 'chr12',
        'CM000675.2': 'chr13',
        'CM000676.2': 'chr14',
        'CM000677.2': 'chr15',
        'CM000678.2': 'chr16',
        'CM000679.2': 'chr17',
        'CM000680.2': 'chr18',
        'CM000681.2': 'chr19',
        'CM000682.2': 'chr20',
        'CM000683.2': 'chr21',
        'CM000684.2': 'chr22',
        'CM000685.2': 'chrX',
        'CM000686.2': 'chrY',
    }
    chrom = chrom.split()[0]  # ensure no spaces
    if chrom in refseq_map:
        return refseq_map[chrom]
    if chrom in genbank_map:
        return genbank_map[chrom]
    return chrom

def convert_trf_dat(input_file, output_file):
    """
    Parses TRF -ngs format (.dat) and converts to BED.
    0-indexed start. Columns: start, end, period, motif
    """
    count = 0
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        chrom = "unknown"
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if line.startswith('@'):
                # Seq header
                chrom = clean_chrom(line[1:].split()[0])
                continue
            parts = line.split()
            if len(parts) >= 15 and parts[0].isdigit():
                try:
                    # TRF is 1-indexed, end is inclusive
                    start = int(parts[0]) - 1
                    end = int(parts[1])
                    period = parts[2]
                    motif = parts[14] if len(parts) > 14 else parts[13] # Sometimes consensus vs actual
                    fout.write(f"{chrom}\t{start}\t{end}\t{motif}\t{period}\tTRF\n")
                    count += 1
                except ValueError:
                    pass
    return count

def convert_mreps_txt(input_file, output_file):
    """
    Parses mreps .txt output.
    """
    count = 0
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        chrom = "unknown"
        in_results = False
        for line in fin:
            line = line.strip()
            if line.startswith("Processing sequence"):
                chrom = clean_chrom(line.split("'")[1].split()[0])
            elif "->" in line and ":" in line and "<" in line:
                try:
                    # Format: from -> to : size <period> ...
                    left, right = line.split(":")
                    start_str, end_str = left.split("->")
                    start = int(start_str.strip()) - 1
                    end = int(end_str.strip())
                    
                    period_str = right.split("<")[1].split(">")[0]
                    period = period_str.strip()
                    
                    motif = "N/A"
                    if "]" in right:
                        right_tokens = right.split("]")[1].split()
                        if len(right_tokens) >= 2:
                            motif = "".join(right_tokens[1:])
                    
                    fout.write(f"{chrom}\t{start}\t{end}\t{motif}\t{period}\tmreps\n")
                    count += 1
                except (ValueError, IndexError):
                    pass
    return count

def convert_mreps_bed(input_file, output_file):
    """
    Parses mreps .bed output format.
    Columns: chrom (with spaces), start, end, name, size, strand, 1-indexed start, 1-indexed end, period, copies, error_rate, motif
    """
    count = 0
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            parts = line.strip().split('\t')
            if len(parts) >= 12:
                chrom = clean_chrom(parts[0].split()[0])
                start = parts[1]
                end = parts[2]
                period = parts[8]
                motif = parts[11]
                fout.write(f"{chrom}\t{start}\t{end}\t{motif}\t{period}\tmreps\n")
                count += 1
    return count

def convert_mreps(input_file, output_file):
    if input_file.endswith('.bed'):
        return convert_mreps_bed(input_file, output_file)
    else:
        return convert_mreps_txt(input_file, output_file)

def convert_ultra_tsv(input_file, output_file):
    """
    Parses ULTRA .tsv output.
    SeqID, Start, End, Period, Score, Consensus
    """
    count = 0
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        reader = csv.reader(fin, delimiter='\t')
        header = next(reader, None) # skip header
        for row in reader:
            if len(row) >= 6:
                chrom = clean_chrom(row[0])
                start = row[1]
                end = row[2]
                period = row[3]
                motif = row[5]
                fout.write(f"{chrom}\t{start}\t{end}\t{motif}\t{period}\tULTRA\n")
                count += 1
    return count

def convert_bwtandem_bed(input_file, output_file):
    """
    Parses BWTandem BED output.
    chrom, start, end, motif, copies, period/tier, mismatch, strand
    """
    count = 0
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                chrom = clean_chrom(parts[0])
                start = parts[1]
                end = parts[2]
                motif = parts[3]
                period = parts[5]
                if period == 'satellite':
                    period = str(len(motif))
                fout.write(f"{chrom}\t{start}\t{end}\t{motif}\t{period}\tBWTandem\n")
                count += 1
    return count

def convert_tantan_bed(input_file, output_file):
    """
    Parses tantan BED format (-f4).
    chrom, start, end, period, copies, consensus
    """
    count = 0
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                chrom = clean_chrom(parts[0])
                start = parts[1]
                end = parts[2]
                period = parts[3]
                motif = parts[5] if len(parts) > 5 else "N/A"
                fout.write(f"{chrom}\t{start}\t{end}\t{motif}\t{period}\ttantan\n")
                count += 1
    return count

def convert_ncrf(input_file, output_file):
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
                chrom = clean_chrom(parts[0])
                range_parts = parts[3].split('-')
                start = range_parts[0]
                end = range_parts[1]
                sequence = parts[4].replace('-', '')
            elif len(parts) >= 4 and 'score=' in parts[2]:
                motif_strand = parts[0]
                strand = motif_strand[-1] if motif_strand[-1] in ('+', '-') else '.'
                score = parts[2].replace('score=', '')
                fout.write(f"{chrom}\t{start}\t{end}\t{sequence}\t{score}\t{strand}\n")
                count += 1
                
    return count

def convert_trash_csv(input_file, output_file):
    """
    Parses TRASH Summary CSV.
    """
    count = 0
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            try:
                chrom = clean_chrom(row.get('name') or row.get('fasta.name', 'unknown'))
                start = row.get('start')
                end = row.get('end')
                motif = row.get('consensus.primary', 'N/A')
                # length of motif is period
                period = str(len(motif)) if motif != 'N/A' and motif != 'none_identified' else 'N/A'
                fout.write(f"{chrom}\t{start}\t{end}\t{motif}\t{period}\tTRASH\n")
                count += 1
            except KeyError:
                pass
    return count

def convert_blast_outfmt6(input_file, output_file, fasta_file=None):
    """
    Parses BLAST outfmt 6 format.
    qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore
    """
    fasta_dict = {}
    if fasta_file and os.path.exists(fasta_file):
        print(f"Loading FASTA {fasta_file} for sequence extraction...")
        with open(fasta_file, 'r') as f:
            header = ""
            seqs = []
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if header:
                        fasta_dict[header] = "".join(seqs)
                    header = clean_chrom(line[1:].split()[0])
                    seqs = []
                else:
                    seqs.append(line)
            if header:
                fasta_dict[header] = "".join(seqs)

    count = 0
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            parts = line.strip().split('\t')
            if len(parts) >= 12:
                try:
                    chrom = clean_chrom(parts[1]) # sseqid
                    sstart = int(parts[8])
                    send = int(parts[9])
                    start = min(sstart, send) - 1 # 0-indexed BED
                    end = max(sstart, send)
                    
                    motif = parts[0] # qseqid default
                    if fasta_dict and chrom in fasta_dict:
                        motif = fasta_dict[chrom][start:end]
                        if sstart > send:
                            trans = str.maketrans("ATCGNatcgn", "TAGCNtagcn")
                            motif = motif.translate(trans)[::-1]
                            
                    period = "."
                    fout.write(f"{chrom}\t{start}\t{end}\t{motif}\t{period}\t.\n")
                    count += 1
                except ValueError:
                    pass
    return count

PARSERS = {
    'trf': convert_trf_dat,
    'mreps': convert_mreps,
    'ultra': convert_ultra_tsv,
    'bwtandem': convert_bwtandem_bed,
    'tantan': convert_tantan_bed,
    'ncrf': convert_ncrf,
    'trash': convert_trash_csv,
    'blast': convert_blast_outfmt6,
}

def merge_beds(input_files, output_file):
    """Concatenates multiple BED files into one, keeping only unique lines."""
    seen = set()
    count = 0
    with open(output_file, 'w') as fout:
        for f in input_files:
            if not os.path.exists(f):
                continue
            with open(f, 'r') as fin:
                for line in fin:
                    if line not in seen:
                        fout.write(line)
                        seen.add(line)
                        count += 1
    print(f"Merged {len(input_files)} files into {output_file} ({count} regions).")

def merge_trash_files(output_dir):
    trash_dir = os.path.join(output_dir, 'trash')
    if not os.path.isdir(trash_dir):
        return
        
    print("Post-processing: Merging TRASH results by genome/experiment...")
    
    # 1. Merge Col-CEN runs
    colcen_files = [
        os.path.join(trash_dir, 'Col-CEN_v1.2_trash_CEN159.bed'),
        os.path.join(trash_dir, 'Col-CEN_v1.2_trash_CEN178.bed'),
        os.path.join(trash_dir, 'Col-CEN_v1.2_trash_denovo.bed')
    ]
    colcen_out = os.path.join(trash_dir, 'Col-CEN_v1.2_trash.bed')
    merge_beds(colcen_files, colcen_out)
    
    # 2. Merge Mo17 exp3A
    mo17_3a_files = [
        os.path.join(trash_dir, 'GCA_022117705.1_Zm-Mo17-REFERENCE-CAU-T2T-assembly_genomic_trash_denovo_exp3A_microsatellite.bed'),
        os.path.join(trash_dir, 'GCA_022117705.1_Zm-Mo17-REFERENCE-CAU-T2T-assembly_genomic_trash_templates_TAG_exp3A_microsatellite.bed')
    ]
    mo17_3a_out = os.path.join(trash_dir, 'GCA_022117705.1_Zm-Mo17-REFERENCE-CAU-T2T-assembly_genomic_trash_exp3A.bed')
    merge_beds(mo17_3a_files, mo17_3a_out)
    
    # 3. Merge Mo17 exp3B
    mo17_3b_files = [
        os.path.join(trash_dir, 'GCA_022117705.1_Zm-Mo17-REFERENCE-CAU-T2T-assembly_genomic_trash_templates_TR1_exp3B_satellite.bed'),
        os.path.join(trash_dir, 'GCA_022117705.1_Zm-Mo17-REFERENCE-CAU-T2T-assembly_genomic_trash_templates_knob180_exp3B_satellite.bed')
    ]
    mo17_3b_out = os.path.join(trash_dir, 'GCA_022117705.1_Zm-Mo17-REFERENCE-CAU-T2T-assembly_genomic_trash_exp3B.bed')
    merge_beds(mo17_3b_files, mo17_3b_out)
    
    # 4. Copy/rename Mo17 exp3C
    mo17_3c_in = os.path.join(trash_dir, 'GCA_022117705.1_Zm-Mo17-REFERENCE-CAU-T2T-assembly_genomic_trash_templates_CentC_exp3C_centc.bed')
    mo17_3c_out = os.path.join(trash_dir, 'GCA_022117705.1_Zm-Mo17-REFERENCE-CAU-T2T-assembly_genomic_trash_exp3C.bed')
    if os.path.exists(mo17_3c_in):
        import shutil
        shutil.copy2(mo17_3c_in, mo17_3c_out)
        print(f"Copied {mo17_3c_in} to {mo17_3c_out}")

def main():
    parser = argparse.ArgumentParser(description="Convert TR outputs to standardized BED.")
    parser.add_argument('--tool', choices=PARSERS.keys(), help='Tool name')
    parser.add_argument('--input', help='Input file')
    parser.add_argument('--output', help='Output BED file')
    parser.add_argument('--batch', action='store_true', help='Run in batch mode over results directory')
    parser.add_argument('--results-dir', default='results', help='Base results directory')
    parser.add_argument('--output-dir', default='beds', help='Base output directory for converted BEDs')
    parser.add_argument('--blast-result', default='results.txt', help='Path to blast results.txt to be converted during batch mode')
    parser.add_argument('--fasta', help='Path to genome FASTA file to extract sequences for blast results')


    args = parser.parse_args()

    if args.batch:
        import shutil
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        
        for tool, func in PARSERS.items():
            tool_res_dir = os.path.join(args.results_dir, tool, 'results')
            tool_out_dir = os.path.join(args.output_dir, tool)
            if not os.path.isdir(tool_res_dir):
                continue
                
            os.makedirs(tool_out_dir, exist_ok=True)
            
            for root, _, files in os.walk(tool_res_dir):
                for f in files:
                    # Exclude log or other unrelated files if any
                    if f.endswith('.log') or f.endswith('.sh') or f.endswith('.settings') or (tool == 'ncrf' and f.endswith('.bed')):
                        continue
                        
                    input_path = os.path.join(root, f)
                    
                    # For trash, name the output BED file using the subdirectory name to avoid collisions
                    if tool == 'trash' and root != tool_res_dir:
                        out_filename = os.path.basename(root) + '.bed'
                    else:
                        if f.endswith('.bed'):
                            out_filename = f
                        elif f.endswith(('.txt', '.dat', '.tsv', '.csv', '.ncrf')):
                            out_filename = f.rsplit('.', 1)[0] + '.bed'
                        else:
                            out_filename = f + '.bed'
                        
                    output_path = os.path.join(tool_out_dir, out_filename)
                    
                    if f.endswith('.bed') and tool not in ('mreps', 'bwtandem', 'tantan'):
                        if os.path.lexists(output_path):
                            os.remove(output_path)
                        os.symlink(os.path.abspath(input_path), output_path)
                        print(f"Symlinked {input_path} -> {output_path}")
                    else:
                        print(f"Converting {input_path} -> {output_path}")
                        count = func(input_path, output_path)
                        print(f"  Done: {count} regions.")
        
        # Merge TRASH results
        merge_trash_files(args.output_dir)

        # Convert blast result if it exists
        blast_in = args.blast_result
        if os.path.exists(blast_in):
            blast_out_dir = os.path.join(args.output_dir, 'blast')
            os.makedirs(blast_out_dir, exist_ok=True)
            
            if blast_in.endswith('.bed'):
                blast_out = os.path.join(blast_out_dir, os.path.basename(blast_in))
                if os.path.lexists(blast_out):
                    os.remove(blast_out)
                os.symlink(os.path.abspath(blast_in), blast_out)
                print(f"Symlinked {blast_in} -> {blast_out}")
            else:
                blast_out = os.path.join(blast_out_dir, 'results.bed')
                print(f"Converting {blast_in} -> {blast_out}")
                count = PARSERS['blast'](blast_in, blast_out, fasta_file=args.fasta)
                print(f"  Done: {count} regions.")
    else:
        if not args.tool or not args.input or not args.output:
            print("Error: --tool, --input, --output required if not in --batch mode.")
            sys.exit(1)
        count = PARSERS[args.tool](args.input, args.output)
        print(f"Converted {count} regions.")

if __name__ == '__main__':
    main()
