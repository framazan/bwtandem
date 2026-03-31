"""Build C extensions for bwtandem."""
import os
import sys
import ctypes
from ctypes import c_int, c_double, c_char_p, POINTER

_EXT_DIR = os.path.dirname(os.path.abspath(__file__))


def _lib_path(name):
    ext = '.dylib' if sys.platform == 'darwin' else '.so'
    return os.path.join(_EXT_DIR, f'lib{name}{ext}')


def get_lib_path():
    """Get the path to the Tier 1 shared library."""
    return _lib_path('tier1_scan')


def _compile(src_name, lib_name):
    """Compile a single C source to a shared library."""
    import subprocess
    src = os.path.join(_EXT_DIR, src_name)
    out = _lib_path(lib_name)
    cmd = ['gcc', '-O3', '-shared', '-fPIC', '-std=c99', '-o', out, src]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"C extension build failed ({src_name}): {result.stderr}", file=sys.stderr)
        return False
    return True


def build():
    """Compile all C extensions."""
    ok = _compile('tier1_scan.c', 'tier1_scan')
    ok2 = _compile('tier2_accel.c', 'tier2_accel')
    return ok and ok2


def load():
    """Load the Tier 1 C library, building if necessary."""
    lib_path = get_lib_path()
    if not os.path.exists(lib_path):
        if not build():
            return None
    try:
        lib = ctypes.CDLL(lib_path)
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


def load_tier2():
    """Load the Tier 2 C library, building if necessary."""
    lib_path = _lib_path('tier2_accel')
    if not os.path.exists(lib_path):
        if not build():
            return None
    try:
        lib = ctypes.CDLL(lib_path)

        lib.smallest_period_str.restype = c_int
        lib.smallest_period_str.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte), c_int
        ]

        lib.smallest_period_str_approx.restype = c_int
        lib.smallest_period_str_approx.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte), c_int, c_double
        ]

        lib.hamming_distance.restype = c_int
        lib.hamming_distance.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.POINTER(ctypes.c_ubyte),
            c_int
        ]

        lib.batch_process_lcp_candidates.restype = c_int
        lib.batch_process_lcp_candidates.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),  # text
            c_int,                            # n
            ctypes.POINTER(c_int),           # periods
            ctypes.POINTER(c_int),           # seed_positions
            c_int,                            # n_candidates
            c_double,                         # max_mismatch_rate
            c_int,                            # min_copies
            ctypes.POINTER(ctypes.c_ubyte),  # covered_mask
            ctypes.POINTER(c_int),           # out_starts
            ctypes.POINTER(c_int),           # out_ends
            ctypes.POINTER(c_int),           # out_periods
            ctypes.POINTER(c_int),           # out_copies
            c_int,                            # max_results
        ]

        return lib
    except OSError as e:
        print(f"Failed to load Tier 2 C extension: {e}", file=sys.stderr)
        return None


if __name__ == '__main__':
    if build():
        print("All C extensions built successfully.")
    else:
        sys.exit(1)
