import time
import subprocess
import os
import argparse
import sys
from trf_to_bed import convert_trf_ngs_to_bed

def run_and_log_trf():
    parser = argparse.ArgumentParser(description="Run TRF and dynamically convert the output to BED format.")
    
    # Required Positional Argument
    parser.add_argument("fasta", type=str, help="Path to the input FASTA sequence file")
    
    # Optional TRF Parameters (Matching standard defaults)
    parser.add_argument("--match", type=int, default=2, help="Matching weight (Default: 2)")
    parser.add_argument("--mismatch", type=int, default=7, help="Mismatching penalty (Default: 7)")
    parser.add_argument("--delta", type=int, default=7, help="Indel penalty (Default: 7)")
    parser.add_argument("--PM", type=int, default=80, help="Match probability [1-100] (Default: 80)")
    parser.add_argument("--PI", type=int, default=10, help="Indel probability [1-100] (Default: 10)")
    parser.add_argument("--minscore", type=int, default=50, help="Minimum alignment score to report (Default: 50)")
    parser.add_argument("--maxperiod", type=int, default=2000, help="Maximum period size to report (Default: 2000)")
    parser.add_argument("-l", type=int, default=6, help="Maximum TR length expected in millions (Default: 6)")
    
    args = parser.parse_args()
    fa_path = os.path.abspath(args.fasta)
    
    # Determine absolute path to the project 'results' folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(root_dir, "results")
    
    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate dynamic file names unconditionally anchored to the results folder
    base_name = os.path.splitext(os.path.basename(fa_path))[0]
    dat_file = os.path.join(results_dir, f"trf_{base_name}.dat")
    bed_file = os.path.join(results_dir, f"trf_{base_name}.bed")
    log_file = os.path.join(results_dir, f"trf_{base_name}.log")
    
    # Construct the TRF invocation command (Since TRF outputs DAT files to wherever it is executed from, we temporarily switch CWD)
    cmd = f"trf {fa_path} {args.match} {args.mismatch} {args.delta} {args.PM} {args.PI} {args.minscore} {args.maxperiod} -l {args.l} -ngs -h"
    
    print(f"Running TRF against {base_name}... this may take a few minutes.")
    start_time = time.time()
    
    # Run TRF and capture the output
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=results_dir)
    elapsed = time.time() - start_time
    
    # Step 1: Temporarily write the stdout payload to a proxy DAT file
    with open(dat_file, "w") as f:
        f.write(result.stdout)
        
    # Step 2: Extract counts and immediately convert the file to BED formatting
    repeat_count = convert_trf_ngs_to_bed(dat_file, bed_file)

    # Step 3: Delete the bulky raw .dat file now that we generated the minimal BED
    if os.path.exists(dat_file):
        os.remove(dat_file)
            
    # Format the log file for validation
    invocation = "python " + " ".join(sys.argv)
    log_content = [
        f"Parent Command: {invocation}",
        f"TRF Command: {cmd}",
        f"",
        f"Processing sequence: {os.path.basename(fa_path)}",
        f"  [TRF] Running Core Engine ({args.match} {args.mismatch} {args.delta} {args.PM} {args.PI} {args.minscore} {args.maxperiod})...",
        f"  [TRF] STDERR (if any): {result.stderr.strip() if result.stderr.strip() else 'None'}",
        f"Total repeats found: {repeat_count}",
        f"Total time: {elapsed:.2f}s",
        f"Results converted and written to -> {bed_file}"
    ]
    
    log_text = "\n".join(log_content) + "\n"
    with open(log_file, "w") as f:
        f.write(log_text)
        
    print(log_text)

if __name__ == "__main__":
    run_and_log_trf()
