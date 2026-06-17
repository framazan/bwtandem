#!/usr/bin/env python3
import pandas as pd
import os

# Chromosome mapping from Mo17 paper (chr1-10) to GenBank (CM039150.1 - CM039159.1)
CHROM_MAP = {
    f"chr{i}": f"CM0391{50 + i - 1}.1" for i in range(1, 11)
}

def extract_arrays(excel_path, sheet_name, out_bed_path, array_name):
    print(f"Extracting {array_name} from sheet {sheet_name}...")
    df = pd.read_excel(excel_path, sheet_name=sheet_name, skiprows=1)
    
    # Drop rows where Chr is NaN
    df = df.dropna(subset=['Chr'])
    
    with open(out_bed_path, 'w') as f:
        count = 0
        for _, row in df.iterrows():
            chr_name = str(row['Chr']).strip()
            # Handle possible variations in chr name
            if chr_name.lower().startswith('chr'):
                chr_name = chr_name.lower()
                
            mapped_chr = CHROM_MAP.get(chr_name, chr_name)
            
            # Start and End might have spaces in column names
            # Find the columns that contain "Start" and "End"
            start_col = [c for c in df.columns if 'Start' in str(c)][0]
            end_col = [c for c in df.columns if 'End' in str(c)][0]
            
            start = int(row[start_col])
            end = int(row[end_col])
            
            # Write BED format (0-indexed start)
            f.write(f"{mapped_chr}\t{start-1}\t{end}\t{array_name}\n")
            count += 1
            
    print(f"  -> Wrote {count} regions to {out_bed_path}")

def main():
    os.makedirs('ground_truth', exist_ok=True)
    excel_file = '41588_2023_1419_MOESM4_ESM.xlsx'
    
    if not os.path.exists(excel_file):
        print(f"Error: {excel_file} not found.")
        return
        
    extract_arrays(excel_file, '6', 'ground_truth/mo17_tr1_arrays.bed', 'TR-1')
    extract_arrays(excel_file, '7', 'ground_truth/mo17_knob180_arrays.bed', 'knob180')
    extract_arrays(excel_file, '8', 'ground_truth/mo17_centc_arrays.bed', 'CentC')
    
    print("Mo17 ground truth extraction complete.")

if __name__ == '__main__':
    main()
