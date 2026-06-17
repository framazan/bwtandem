#!/usr/bin/env python3
import os
import glob
import pandas as pd

def parse_time_str(time_str):
    """Parses h:mm:ss or m:ss into total seconds"""
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = parts
        # sometimes s can be a float like 12.34
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(time_str)

def parse_log_file(filepath):
    wall_clock = 0.0
    max_rss_gb = 0.0
    cpu_percent = "0%"
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if "Elapsed (wall clock) time" in line:
                time_str = line.split("): ")[1].strip()
                wall_clock = parse_time_str(time_str)
            elif "Maximum resident set size" in line:
                kb = line.split("): ")[1].strip()
                max_rss_gb = float(kb) / (1024 * 1024)
            elif "Percent of CPU this job got" in line:
                cpu_percent = line.split(": ")[1].strip()
                
    return wall_clock, max_rss_gb, cpu_percent

def main():
    results_dir = 'results'
    tools = ['trf', 'mreps', 'ultra', 'bwtandem', 'tantan', 'ncrf', 'trash']
    
    data = []
    
    for tool in tools:
        log_dir = os.path.join(results_dir, tool, 'logs')
        if not os.path.isdir(log_dir):
            continue
            
        for log_file in glob.glob(os.path.join(log_dir, '*.log')):
            filename = os.path.basename(log_file)
            # Extrapolate genome/experiment from filename
            # Filenames usually look like: Col-CEN_v1.2_run.log or GCA_000001405.15_GRCh38_genomic_run.log
            # or GCA_022117705.1_Zm-Mo17-REFERENCE-CAU-T2T-assembly_genomic_trf_exp3A_microsatellite_run.log
            
            exp = "unknown"
            genome = "unknown"
            
            if "Col-CEN" in filename:
                exp = "Exp2"
                genome = "Col-CEN"
            elif "GRCh38" in filename:
                exp = "Exp1"
                genome = "GRCh38"
                if "GCF" in filename:
                    # Distinguish between GCA and GCF just in case
                    genome = "GRCh38 (GCF)"
                elif "GCA" in filename:
                    genome = "GRCh38 (GCA)"
            elif "Mo17" in filename:
                genome = "Mo17"
                if "exp3A" in filename.lower() or "microsatellite" in filename.lower():
                    exp = "Exp3A"
                elif "exp3B" in filename.lower() or "satellite" in filename.lower():
                    exp = "Exp3B"
                elif "exp3C" in filename.lower() or "centc" in filename.lower():
                    exp = "Exp3C"
            
            wall, rss, cpu = parse_log_file(log_file)
            data.append({
                'Tool': tool,
                'Genome': genome,
                'Experiment': exp,
                'Wall_Time_s': round(wall, 2),
                'Max_RSS_GB': round(rss, 2),
                'CPU_Percent': cpu
            })
            
    df = pd.DataFrame(data)
    os.makedirs('reports', exist_ok=True)
    out_csv = 'reports/runtime_memory.csv'
    df.to_csv(out_csv, index=False)
    print(f"Parsed {len(df)} logs. Saved to {out_csv}")
    
    # Generate simple markdown table
    md_out = 'reports/runtime_memory.md'
    with open(md_out, 'w') as f:
        f.write("# Runtime and Memory Usage\n\n")
        
        for exp in sorted(df['Experiment'].unique()):
            f.write(f"## {exp}\n\n")
            exp_df = df[df['Experiment'] == exp].copy()
            exp_df = exp_df.sort_values('Wall_Time_s')
            f.write(exp_df.to_markdown(index=False))
            f.write("\n\n")
            
    print(f"Markdown report generated at {md_out}")

if __name__ == '__main__':
    main()
