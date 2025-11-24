import sys
import os
import time
import contextlib
import io
from typing import List, Dict, Any
import numpy as np

# Add root to path to import bwt.py
sys.path.append(os.getcwd())

# Import Old Implementation
try:
    import bwt as old_bwt
except ImportError:
    print("Error: Could not import bwt.py. Make sure you are in the root directory.")
    sys.exit(1)

# Import New Implementation
try:
    from src.finder import TandemRepeatFinder as NewFinder
    from src.main import parse_fasta
except ImportError:
    print("Error: Could not import src.finder. Make sure src/ exists.")
    sys.exit(1)

@contextlib.contextmanager
def suppress_stdout():
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        yield

def run_old_implementation(fasta_path: str) -> Dict[str, Any]:
    """Run the old bwt.py implementation."""
    start_time = time.time()
    
    # Initialize with flank_trim=0 to match new implementation
    # and other parameters to match defaults
    finder = old_bwt.TandemRepeatFinder(
        reference_file=fasta_path,
        flank_trim=0,
        show_progress=False
    )
    
    # Load and build
    sequences = finder.load_reference()
    finder.build_indices(sequences)
    
    # Find
    repeats = finder.find_tandem_repeats()
    
    # Post-processing (mimicking bwt.py's main block if it had one, 
    # but find_tandem_repeats seems to return raw-ish repeats, 
    # actually it calls _deduplicate_repeats internally? 
    # No, find_tandem_repeats in bwt.py calls _deduplicate_repeats? 
    # Let's check the code. It seems it does NOT call deduplicate/merge/refine 
    # inside find_tandem_repeats in the snippet I saw. 
    # Wait, I need to check if bwt.py does post-processing inside find_tandem_repeats.
    # If not, I should call them to be fair.)
    
    # Checking bwt.py source again...
    # It seems find_tandem_repeats just collects them.
    # The main execution block usually does the post-processing.
    # I will apply the same post-processing as the script likely does.
    
    repeats = finder._deduplicate_repeats(repeats)
    repeats = finder._merge_adjacent_repeats(repeats)
    repeats = finder._refine_repeats(repeats)
    
    end_time = time.time()
    
    return {
        "repeats": repeats,
        "time": end_time - start_time,
        "count": len(repeats)
    }

def run_new_implementation(fasta_path: str) -> Dict[str, Any]:
    """Run the new src/ implementation."""
    start_time = time.time()
    
    sequences = parse_fasta(fasta_path)
    all_repeats = []
    
    for chrom, seq in sequences:
        # Match bwt.py default min_period=10
        finder = NewFinder(seq, chromosome=chrom, min_period=10, show_progress=False)
        repeats = finder.find_all()
        all_repeats.extend(repeats)
        finder.cleanup()
        
    end_time = time.time()
    
    return {
        "repeats": all_repeats,
        "time": end_time - start_time,
        "count": len(all_repeats)
    }

def compare_results(old_res, new_res, name):
    print(f"\n--- Comparison for {name} ---")
    print(f"{'Metric':<20} | {'Old (bwt.py)':<15} | {'New (src/)':<15} | {'Diff':<10}")
    print("-" * 70)
    
    t_old = old_res['time']
    t_new = new_res['time']
    t_diff = (t_new - t_old) / t_old * 100 if t_old > 0 else 0
    print(f"{'Time (s)':<20} | {t_old:<15.4f} | {t_new:<15.4f} | {t_diff:+.1f}%")
    
    c_old = old_res['count']
    c_new = new_res['count']
    c_diff = c_new - c_old
    print(f"{'Repeat Count':<20} | {c_old:<15} | {c_new:<15} | {c_diff:+}")
    
    # Detailed comparison
    # We'll check how many old repeats are "covered" by new repeats
    covered_count = 0
    exact_matches = 0
    
    # Create a simple interval tree or just brute force for small N
    new_intervals = {} # chrom -> list of (start, end, motif)
    for r in new_res['repeats']:
        if r.chrom not in new_intervals:
            new_intervals[r.chrom] = []
        new_intervals[r.chrom].append((r.start, r.end, r.motif))
        
    for r_old in old_res['repeats']:
        chrom = r_old.chrom
        if chrom in new_intervals:
            found = False
            for start, end, motif in new_intervals[chrom]:
                # Check overlap
                overlap_start = max(r_old.start, start)
                overlap_end = min(r_old.end, end)
                if overlap_end > overlap_start:
                    # Significant overlap?
                    overlap_len = overlap_end - overlap_start
                    union_len = (max(r_old.end, end) - min(r_old.start, start))
                    if overlap_len / union_len > 0.5:
                        found = True
                        if r_old.start == start and r_old.end == end and r_old.motif == motif:
                            exact_matches += 1
                        break
            if found:
                covered_count += 1
                
    print(f"{'Covered (Old in New)':<20} | {covered_count}/{c_old} ({covered_count/c_old*100:.1f}%)")
    print(f"{'Exact Matches':<20} | {exact_matches}/{c_old} ({exact_matches/c_old*100:.1f}%)")

def main():
    test_files = [
        ("Short/Mixed", "test2.fa"),
        # ("Synthetic", "test_synthetic.fasta"), # Uncomment if exists
    ]
    
    # Check if synthetic exists
    if os.path.exists("test_synthetic.fasta"):
        test_files.append(("Synthetic", "test_synthetic.fasta"))
        
    # Create a long random sequence for stress test
    long_file = "test_long_random.fa"
    if not os.path.exists(long_file):
        print("Generating long random sequence...")
        with open(long_file, "w") as f:
            f.write(">long_random\n")
            # 100k random bases
            seq = "".join(np.random.choice(list("ACGT"), 100000))
            f.write(seq + "\n")
    test_files.append(("Long Random (100kb)", long_file))

    for name, path in test_files:
        if not os.path.exists(path):
            print(f"Skipping {name}: {path} not found")
            continue
            
        print(f"\nRunning {name} ({path})...")
        
        # Run Old
        with suppress_stdout():
            old_res = run_old_implementation(path)
            
        # Run New
        with suppress_stdout():
            new_res = run_new_implementation(path)
            
        compare_results(old_res, new_res, name)

if __name__ == "__main__":
    main()
