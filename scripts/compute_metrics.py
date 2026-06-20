#!/usr/bin/env python3
import argparse
import os
import glob
import pandas as pd
import subprocess
import tempfile

def run_bedtools(cmd):
    """Run a bedtools command and return stdout as string."""
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
    """Percent of ground truth regions that overlap with tool calls (Recall)"""
    stdout = run_bedtools(["bedtools", "intersect", "-a", gt_bed, "-b", bed_file, "-wa", "-u"])
    overlap_count = len([x for x in stdout.split('\n') if x])
    
    with open(gt_bed) as f:
        total_gt = sum(1 for line in f if line.strip())
        
    if total_gt == 0:
        return 0
    return (overlap_count / total_gt) * 100

def compute_precision(bed_file, gt_bed):
    """Percent of tool calls that overlap with a ground truth region."""
    stdout = run_bedtools(["bedtools", "intersect", "-a", bed_file, "-b", gt_bed, "-wa", "-u"])
    overlap_count = len([x for x in stdout.split('\n') if x])
    regions, _ = total_regions_and_bp(bed_file)
    if regions == 0:
        return 0
    return (overlap_count / regions) * 100

def compute_bp_coverage(bed_file, gt_bed):
    """Calculate the fraction of base pairs intersecting (Recall) and the precision."""
    # First sort the bed file
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

def compute_boundary_offset(bed_file, array_bed):
    """Mean absolute distance between predicted array boundaries and true array boundaries."""
    # Find number of columns in array_bed to parse the -wa -wb output correctly
    try:
        with open(array_bed) as f:
            a_cols = len(f.readline().strip().split('\t'))
    except Exception:
        return 0
        
    stdout = run_bedtools(["bedtools", "intersect", "-a", array_bed, "-b", bed_file, "-wa", "-wb"])
    if not stdout.strip():
        return 0
    
    array_matches = {}
    for line in stdout.split('\n'):
        if not line.strip(): continue
        parts = line.split('\t')
        gt_key = (parts[0], int(parts[1]), int(parts[2]))
        t_start = int(parts[a_cols + 1])
        t_end = int(parts[a_cols + 2])
        if gt_key not in array_matches:
            array_matches[gt_key] = []
        array_matches[gt_key].append((t_start, t_end))
        
    offsets = []
    for gt_key, t_spans in array_matches.items():
        gt_start, gt_end = gt_key[1], gt_key[2]
        min_t_start = min(s for s, e in t_spans)
        max_t_end = max(e for s, e in t_spans)
        offset = (abs(gt_start - min_t_start) + abs(gt_end - max_t_end)) / 2
        offsets.append(offset)
        
    return sum(offsets) / len(offsets) if offsets else 0

def compute_centromere_coverage(bed_file, cen_bed):
    stdout = run_bedtools(["bedtools", "coverage", "-a", cen_bed, "-b", bed_file])
    coverages = []
    for line in stdout.split('\n'):
        if line.strip():
            parts = line.split('\t')
            frac = float(parts[-1])
            coverages.append(frac)
    if not coverages:
        return 0
    return (sum(coverages) / len(coverages)) * 100

def compute_cen180_count(bed_file, cen_bed):
    stdout = run_bedtools(["bedtools", "intersect", "-a", bed_file, "-b", cen_bed, "-wa", "-u"])
    count = 0
    for line in stdout.split('\n'):
        if line.strip():
            parts = line.split('\t')
            if len(parts) >= 5:
                try:
                    period = float(parts[4])
                    if 150 <= period <= 200: 
                        count += 1
                except ValueError:
                    pass
    return count

def compute_array_detection(bed_file, array_bed):
    stdout = run_bedtools(["bedtools", "intersect", "-a", array_bed, "-b", bed_file, "-wa", "-u"])
    return len([x for x in stdout.split('\n') if x])

def compute_fragmentation(bed_file, array_bed):
    stdout = run_bedtools(["bedtools", "intersect", "-a", array_bed, "-b", bed_file, "-c"])
    fragments = []
    for line in stdout.split('\n'):
        if line.strip():
            parts = line.split('\t')
            count = int(parts[-1])
            if count > 0:
                fragments.append(count)
    if not fragments:
        return 0
    return sum(fragments) / len(fragments)

def compute_norm_fragmentation(bed_file, array_bed):
    frag = compute_fragmentation(bed_file, array_bed)
    return round(1.0 / frag, 2) if frag > 0 else 0

def compute_tag_metrics(bed_file):
    tag_count = 0
    longest_tag = 0
    try:
        with open(bed_file, 'r') as f:
            for line in f:
                if not line.strip(): continue
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    start = int(parts[1])
                    end = int(parts[2])
                    motif_or_seq = parts[3].upper()
                    if 'TAG' in motif_or_seq or 'CTA' in motif_or_seq:
                        tag_count += 1
                        length = end - start
                        if length > longest_tag:
                            longest_tag = length
    except Exception:
        pass
    return tag_count, longest_tag / 1000.0

def get_bed_files(tools, genome, exp_code, base_dir='beds', must_include=None):
    files = {}
    for tool_alias in tools:
        if tool_alias.startswith('trash'):
            tool_dir = os.path.join(base_dir, 'trash')
        else:
            tool_dir = os.path.join(base_dir, tool_alias)
            
        if not os.path.exists(tool_dir):
            continue
        
        matches = []
        for f in glob.glob(os.path.join(tool_dir, f'*{genome}*{exp_code}*.bed')):
            matches.append(f)
        if not matches:
            for f in glob.glob(os.path.join(tool_dir, f'*{genome}*.bed')):
                matches.append(f)
        
        if must_include:
            matches = [m for m in matches if must_include in m]
            
        # Additional logic for trash aliases
        if tool_alias == 'trash_denovo':
            # Remove anything that is explicitly a template run
            matches = [m for m in matches if 'template' not in m.lower() and 'CEN1' not in m and 'CentC' not in m and 'TAG' not in m and 'TR1' not in m and 'knob180' not in m]
            # Prioritize files explicitly named 'denovo'
            matches.sort(key=lambda x: 0 if 'denovo' in x.lower() else 1)
        elif tool_alias == 'trash_template':
            # Remove anything that is explicitly a denovo run
            matches = [m for m in matches if 'denovo' not in m.lower()]
            # Prioritize files explicitly named 'template' or the base '_trash.bed'
            matches.sort(key=lambda x: 0 if ('template' in x.lower() or x.endswith('_trash.bed')) else 1)

        if matches:
            files[tool_alias] = matches[0] # Just take the first valid match
    return files

def get_runtime_memory(df_rm, tool, genome_like, exp_like):
    if df_rm is None:
        return "N/A", "N/A"
    
    # Map back aliases to base tool for runtime matching
    tool_base = tool.split('_')[0] if tool.startswith('trash') else tool
    
    for _, row in df_rm.iterrows():
        r_tool = str(row['Tool'])
        r_genome = str(row['Genome'])
        r_exp = str(row['Experiment'])
        if r_tool == tool_base and genome_like in r_genome and exp_like in r_exp:
            return str(row['Wall_Time_s']), str(row['Max_RSS_GB'])
    return "N/A", "N/A"

def compute_unique(tool_bed, other_beds):
    if not other_beds:
        return total_regions_and_bp(tool_bed)[0]
    cmd = ["bedtools", "intersect", "-a", tool_bed, "-b"] + other_beds + ["-v"]
    stdout = run_bedtools(cmd)
    return len([x for x in stdout.split('\n') if x])

def main():
    tools = ['trf', 'mreps', 'ultra', 'bwtandem', 'tantan', 'ncrf', 'trash_denovo', 'trash_template']
    os.makedirs('reports', exist_ok=True)
    
    try:
        if os.path.exists('benchmarking/reports/runtime_memory.csv'):
            df_rm = pd.read_csv('benchmarking/reports/runtime_memory.csv')
        elif os.path.exists('reports/runtime_memory.csv'):
            df_rm = pd.read_csv('reports/runtime_memory.csv')
        else:
            df_rm = None
    except Exception:
        df_rm = None

    # ---------------- EXPERIMENT 1 ----------------
    print("Running Experiment 1: Human GRCh38")
    bed_files_exp1 = get_bed_files(tools, "GRCh38", "", must_include="GCF")
    adotto_bed = "ground_truth/adotto_tr_regions.bed"
    
    data_exp1 = []
    for tool, bed in bed_files_exp1.items():
        regions, bp = total_regions_and_bp(bed)
        overlap_recall = compute_overlap(bed, adotto_bed) if os.path.exists(adotto_bed) else 0
        overlap_precision = compute_precision(bed, adotto_bed) if os.path.exists(adotto_bed) else 0
        bp_recall, bp_precision = compute_bp_coverage(bed, adotto_bed) if os.path.exists(adotto_bed) else (0,0)
        
        # Unique regions
        other_beds = [b for t, b in bed_files_exp1.items() if t != tool]
        unique_count = compute_unique(bed, other_beds) if other_beds else regions
        
        runtime, memory = get_runtime_memory(df_rm, tool, "GRCh38", "Exp1")
        
        data_exp1.append({
            'Tool': tool,
            'Total Regions': regions,
            'Adotto Recall (%)': round(overlap_recall, 2),
            'Adotto Precision (%)': round(overlap_precision, 2),
            'BP Recall (%)': round(bp_recall, 2),
            'BP Precision (%)': round(bp_precision, 2),
            'Unique Regions': unique_count,
            'Runtime (s)': runtime,
            'Memory (GB)': memory
        })
    df_exp1 = pd.DataFrame(data_exp1)
    df_exp1.to_csv('reports/experiment1_metrics.csv', index=False)
    if not df_exp1.empty:
        with open('reports/experiment1_metrics.md', 'w') as f:
            f.write(df_exp1.to_markdown(index=False))
    
    # ---------------- EXPERIMENT 2 ----------------
    print("Running Experiment 2: Col-CEN")
    bed_files_exp2 = get_bed_files(tools, "Col-CEN", "")
    cen_bed = "ground_truth/colcen_centromeres.bed"
    
    data_exp2 = []
    for tool, bed in bed_files_exp2.items():
        regions, bp = total_regions_and_bp(bed)
        cen_cov = compute_centromere_coverage(bed, cen_bed) if os.path.exists(cen_bed) else 0
        cen180_count = compute_cen180_count(bed, cen_bed) if os.path.exists(cen_bed) else 0
        runtime, memory = get_runtime_memory(df_rm, tool, "Col-CEN", "Exp2")
        data_exp2.append({
            'Tool': tool,
            'Regions': regions,
            'Centromere Cov (%)': round(cen_cov, 2),
            'CEN180 Count': cen180_count,
            'Runtime (s)': runtime,
            'Memory (GB)': memory
        })
    df_exp2 = pd.DataFrame(data_exp2)
    df_exp2.to_csv('reports/experiment2_metrics.csv', index=False)
    if not df_exp2.empty:
        with open('reports/experiment2_metrics.md', 'w') as f:
            f.write(df_exp2.to_markdown(index=False))
    
    # ---------------- EXPERIMENT 3A, 3B, 3C ----------------
    print("Running Experiment 3: Mo17")
    
    # 3A Microsatellites
    data_exp3a = []
    for tool, bed in get_bed_files(tools, "Mo17", "exp3A").items():
        regions, bp = total_regions_and_bp(bed)
        tag_count, longest_tag = compute_tag_metrics(bed)
        runtime, memory = get_runtime_memory(df_rm, tool, "Mo17", "Exp3A")
        data_exp3a.append({
            'Tool': tool, 
            'Total SSR bp': bp, 
            'Regions': regions, 
            'TAG arrays found': tag_count,
            'longest TAG (kb)': round(longest_tag, 2),
            'Runtime (s)': runtime, 
            'Memory (GB)': memory
        })
    df_exp3a = pd.DataFrame(data_exp3a)
    df_exp3a.to_csv('reports/experiment3a_metrics.csv', index=False)
    if not df_exp3a.empty:
        with open('reports/experiment3a_metrics.md', 'w') as f:
            f.write(df_exp3a.to_markdown(index=False))
    
    # 3B Satellites (knob180/TR-1)
    data_exp3b = []
    knob180_bed = "ground_truth/mo17_knob180_arrays.bed"
    tr1_bed = "ground_truth/mo17_tr1_arrays.bed"
    for tool, bed in get_bed_files(tools, "Mo17", "exp3B").items():
        k_count = compute_array_detection(bed, knob180_bed) if os.path.exists(knob180_bed) else 0
        t_count = compute_array_detection(bed, tr1_bed) if os.path.exists(tr1_bed) else 0
        k_frag = compute_norm_fragmentation(bed, knob180_bed) if os.path.exists(knob180_bed) else 0
        k_off = compute_boundary_offset(bed, knob180_bed) if os.path.exists(knob180_bed) else 0
        t_off = compute_boundary_offset(bed, tr1_bed) if os.path.exists(tr1_bed) else 0
        runtime, memory = get_runtime_memory(df_rm, tool, "Mo17", "Exp3B")
        data_exp3b.append({
            'Tool': tool, 
            'knob180 arrays (of 25)': k_count, 
            'TR-1 arrays (of 17)': t_count,
            'knob180 Norm Frag Score': k_frag,
            'knob180 Mean Offset Error (bp)': round(k_off, 2),
            'TR-1 Mean Offset Error (bp)': round(t_off, 2),
            'Runtime (s)': runtime,
            'Memory (GB)': memory
        })
    df_exp3b = pd.DataFrame(data_exp3b)
    df_exp3b.to_csv('reports/experiment3b_metrics.csv', index=False)
    if not df_exp3b.empty:
        with open('reports/experiment3b_metrics.md', 'w') as f:
            f.write(df_exp3b.to_markdown(index=False))
    
    # 3C CentC
    data_exp3c = []
    centc_bed = "ground_truth/mo17_centc_arrays.bed"
    gt_c_regions, gt_c_bp = total_regions_and_bp(centc_bed) if os.path.exists(centc_bed) else (0, 0)
    for tool, bed in get_bed_files(tools, "Mo17", "exp3C").items():
        c_count = compute_array_detection(bed, centc_bed) if os.path.exists(centc_bed) else 0
        c_frag = compute_norm_fragmentation(bed, centc_bed) if os.path.exists(centc_bed) else 0
        c_off = compute_boundary_offset(bed, centc_bed) if os.path.exists(centc_bed) else 0
        
        # New metrics
        cen_cov = compute_centromere_coverage(bed, centc_bed) if os.path.exists(centc_bed) else 0
        _, tool_bp = total_regions_and_bp(bed)
        
        runtime, memory = get_runtime_memory(df_rm, tool, "Mo17", "Exp3C")
        data_exp3c.append({
            'Tool': tool, 
            'CentC arrays (of 17)': c_count,
            'Total CentC bp (out of)': f"{tool_bp} / {gt_c_bp}",
            'centromere coverage (%)': round(cen_cov, 2),
            'CentC Norm Frag Score': c_frag,
            'CentC Mean Offset Error (bp)': round(c_off, 2),
            'Runtime (s)': runtime,
            'Memory (GB)': memory
        })
    df_exp3c = pd.DataFrame(data_exp3c)
    df_exp3c.to_csv('reports/experiment3c_metrics.csv', index=False)
    if not df_exp3c.empty:
        with open('reports/experiment3c_metrics.md', 'w') as f:
            f.write(df_exp3c.to_markdown(index=False))

    print("All metrics computed and saved to reports/")

if __name__ == '__main__':
    main()
