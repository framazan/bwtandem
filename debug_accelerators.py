
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from src import accelerators
    print(f"Accelerators loaded: {accelerators}")
    print(f"Native module: {accelerators._native}")
    
    if accelerators._native:
        print("Testing hamming_distance...")
        a = np.array([1, 2, 3], dtype=np.uint8)
        b = np.array([1, 2, 4], dtype=np.uint8)
        dist = accelerators.hamming_distance(a, b)
        print(f"Hamming distance: {dist}")
    else:
        print("Native module not loaded.")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
