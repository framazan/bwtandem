#!/usr/bin/env python3
import os
import tempfile
import subprocess

def run_bedtools(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running: {' '.join(cmd)}")
        print(e.stderr)
        return ""

def total_regions_and_bp(bed_file):
    regions = 0
    bp = 0
    with open(bed_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                start = int(parts[1])
                end = int(parts[2])
                regions += 1
                bp += (end - start)
    return regions, bp

def compute_overlap(bed_file, gt_bed):
    stdout = run_bedtools(["bedtools", "intersect", "-a", gt_bed, "-b", bed_file, "-wa", "-u"])
    overlap_count = len([x for x in stdout.split('\n') if x])
    
    with open(gt_bed) as f:
        total_gt = sum(1 for line in f if line.strip())
        
    if total_gt == 0:
        return 0
    return (overlap_count / total_gt) * 100

def compute_precision(bed_file, gt_bed):
    stdout = run_bedtools(["bedtools", "intersect", "-a", bed_file, "-b", gt_bed, "-wa", "-u"])
    overlap_count = len([x for x in stdout.split('\n') if x])
    regions, _ = total_regions_and_bp(bed_file)
    if regions == 0:
        return 0
    return (overlap_count / regions) * 100

def compute_bp_coverage(bed_file, gt_bed):
    sorted_cmd = ["bedtools", "sort", "-i", bed_file]
    sorted_tool = run_bedtools(sorted_cmd)
    with tempfile.NamedTemporaryFile('w', delete=False) as stf:
        stf.write(sorted_tool)
        temp_sorted = stf.name
        
    merge_cmd = ["bedtools", "merge", "-i", temp_sorted]
    merged_tool = run_bedtools(merge_cmd)
    with tempfile.NamedTemporaryFile('w', delete=False) as tf:
        tf.write(merged_tool)
        temp_tool = tf.name
    
    intersect_cmd = ["bedtools", "intersect", "-a", gt_bed, "-b", temp_tool]
    intersect_out = run_bedtools(intersect_cmd)
    intersect_bp = sum([int(p[2]) - int(p[1]) for line in intersect_out.split('\n') if line.strip() and len(p:=line.split('\t')) >= 3])
    
    gt_regions, gt_bp = total_regions_and_bp(gt_bed)
    tool_regions, tool_bp_merged = total_regions_and_bp(temp_tool)
    
    os.remove(temp_sorted)
    os.remove(temp_tool)
    
    recall = (intersect_bp / gt_bp * 100) if gt_bp > 0 else 0
    precision = (intersect_bp / tool_bp_merged * 100) if tool_bp_merged > 0 else 0
    
    return recall, precision

def map_chromosomes(input_bed):
    temp_bed = tempfile.NamedTemporaryFile('w', delete=False)
    with open(input_bed, 'r') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split('\t')
            chrom = parts[0]
            if chrom.startswith("NC_0000"):
                try:
                    num = int(chrom.split('.')[0].replace("NC_0000", ""))
                    if num <= 22:
                        parts[0] = f"chr{num}"
                    elif num == 23:
                        parts[0] = "chrX"
                    elif num == 24:
                        parts[0] = "chrY"
                except Exception:
                    pass
            elif chrom.startswith("NC_012920"):
                parts[0] = "chrM"
            temp_bed.write('\t'.join(parts) + '\n')
    temp_bed.close()
    return temp_bed.name

def main():
    os.makedirs('reports', exist_ok=True)
    
    bed_file = "results/bwtandem/rerun.bed"
    adotto_bed = "ground_truth/adotto_tr_regions.bed"
    
    print(f"Running metrics on {bed_file} against {adotto_bed}")
    
    if not os.path.exists(bed_file):
        print(f"Error: {bed_file} not found.")
        return
        
    if not os.path.exists(adotto_bed):
        print(f"Error: {adotto_bed} not found.")
        return

    mapped_bed = map_chromosomes(bed_file)

    regions, bp = total_regions_and_bp(mapped_bed)
    overlap_recall = compute_overlap(mapped_bed, adotto_bed)
    overlap_precision = compute_precision(mapped_bed, adotto_bed)
    bp_recall, bp_precision = compute_bp_coverage(mapped_bed, adotto_bed)
    
    overlap_recall = round(overlap_recall, 2)
    overlap_precision = round(overlap_precision, 2)
    bp_recall = round(bp_recall, 2)
    bp_precision = round(bp_precision, 2)
    
    print("\nMetrics:")
    print(f"{'Tool':<20} | {'Total Regions':<15} | {'Adotto Recall (%)':<20} | {'Adotto Precision (%)':<20} | {'BP Recall (%)':<15} | {'BP Precision (%)':<15}")
    print("-" * 115)
    print(f"{'bwtandem_sensitive':<20} | {regions:<15} | {overlap_recall:<20} | {overlap_precision:<20} | {bp_recall:<15} | {bp_precision:<15}")
    
    with open('reports/human_rerun_metrics.csv', 'w') as f:
        f.write("Tool,Total Regions,Adotto Recall (%),Adotto Precision (%),BP Recall (%),BP Precision (%)\n")
        f.write(f"bwtandem_sensitive,{regions},{overlap_recall},{overlap_precision},{bp_recall},{bp_precision}\n")
        
    with open('reports/human_rerun_metrics.md', 'w') as f:
        f.write("| Tool | Total Regions | Adotto Recall (%) | Adotto Precision (%) | BP Recall (%) | BP Precision (%) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        f.write(f"| bwtandem_sensitive | {regions} | {overlap_recall} | {overlap_precision} | {bp_recall} | {bp_precision} |\n")
        
    print("\nSaved to reports/human_rerun_metrics.md and .csv")
    os.remove(mapped_bed)

if __name__ == '__main__':
    main()
