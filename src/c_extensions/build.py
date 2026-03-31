"""Build C extensions for bwtandem."""
import os
import sys
import ctypes
from ctypes import c_int, c_char_p, POINTER

def get_lib_path():
    """Get the path to the compiled shared library."""
    ext_dir = os.path.dirname(os.path.abspath(__file__))
    if sys.platform == 'darwin':
        return os.path.join(ext_dir, 'libtier1_scan.dylib')
    return os.path.join(ext_dir, 'libtier1_scan.so')

def build():
    """Compile C extensions."""
    ext_dir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(ext_dir, 'tier1_scan.c')
    lib_path = get_lib_path()

    # Compile with optimization
    import subprocess
    cmd = [
        'gcc', '-O3', '-shared', '-fPIC', '-std=c99',
        '-o', lib_path, src
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"C extension build failed: {result.stderr}", file=sys.stderr)
        return False
    return True

def load():
    """Load the compiled C library, building if necessary."""
    lib_path = get_lib_path()
    if not os.path.exists(lib_path):
        if not build():
            return None
    try:
        lib = ctypes.CDLL(lib_path)
        # Set up function signature
        lib.find_period_runs.restype = c_int
        lib.find_period_runs.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),  # text
            c_int,                            # n
            c_int,                            # k
            c_int,                            # min_seed_copies
            ctypes.POINTER(ctypes.c_ubyte),  # seen
            ctypes.POINTER(c_int),           # out_starts
            ctypes.POINTER(c_int),           # out_ends
            ctypes.POINTER(c_int),           # out_copies
            c_int,                            # max_candidates
        ]
        return lib
    except OSError as e:
        print(f"Failed to load C extension: {e}", file=sys.stderr)
        return None

if __name__ == '__main__':
    if build():
        print("C extensions built successfully.")
    else:
        sys.exit(1)
