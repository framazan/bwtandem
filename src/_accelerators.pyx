# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Cython accelerators for tandem repeat extension loops."""

import numpy as np
cimport numpy as cnp
from libc.math cimport ceil

ctypedef cnp.uint8_t UINT8

cdef inline int _max_mismatch_threshold(int period, int copies, double allowed_rate):
    """Reproduce Python mismatch threshold logic in Cython."""
    if period <= 0 or copies <= 0:
        return 0
    if period == 1:
        return 0

    if allowed_rate < 0.01:
        allowed_rate = 0.01
    elif allowed_rate > 0.5:
        allowed_rate = 0.5

    cdef double total_len = period * copies
    cdef int threshold = <int>ceil(allowed_rate * total_len)
    if threshold < 1:
        threshold = 1
    return threshold

cdef inline int _total_mismatches(const unsigned char[:] text_arr, int start_pos, int end_pos,
                                  const unsigned char[:] consensus, int period, int n) nogil:
    cdef int copies = (end_pos - start_pos) // period
    cdef int total = 0
    cdef int copy_start, copy_end, idx, pos

    for idx in range(copies):
        copy_start = start_pos + idx * period
        copy_end = copy_start + period
        if copy_end > n:
            break
        for pos in range(period):
            if text_arr[copy_start + pos] != consensus[pos]:
                total += 1
    return total

cdef inline int _hamming_distance(const unsigned char[:] a, const unsigned char[:] b, int length) nogil:
    cdef int i, mismatches = 0
    for i in range(length):
        if a[i] != b[i]:
            mismatches += 1
    return mismatches

cpdef int hamming_distance(const unsigned char[:] arr1, const unsigned char[:] arr2):
    """Calculate Hamming distance between two arrays."""
    cdef int n1 = arr1.shape[0]
    cdef int n2 = arr2.shape[0]
    if n1 != n2:
        raise ValueError("Arrays must have same length")
    
    return _hamming_distance(arr1, arr2, n1)

cpdef tuple extend_with_mismatches(const unsigned char[:] s_arr,
                                   int start_pos, int period, int n,
                                   double allowed_mismatch_rate):
    """Accelerated version of Tier2/Tier1 bidirectional extension.

    Returns (array_start, array_end, copies, full_start, full_end) or None on failure.
    """
    if period <= 0:
        return None

    cdef Py_ssize_t arr_len = s_arr.shape[0]
    if n <= 0 or n > arr_len:
        n = arr_len

    if start_pos < 0 or start_pos + period > n:
        return None

    cdef cnp.ndarray[UINT8, ndim=1] consensus_arr = np.array(
        s_arr[start_pos:start_pos + period], copy=True
    )
    cdef const unsigned char[:] consensus = consensus_arr
    cdef const unsigned char[:] text_arr = s_arr

    cdef int start = start_pos
    cdef int end = start_pos + period
    cdef int copies = 1
    cdef int temp_copies, temp_end, temp_start
    cdef int new_mm, max_mm
    cdef int full_start, full_end
    cdef int array_start, array_end
    cdef int partial_left, partial_right

    # Extend right
    while end + period <= n:
        temp_copies = copies + 1
        temp_end = end + period
        new_mm = _hamming_distance(text_arr[end:end + period], consensus, period)
        max_mm = _max_mismatch_threshold(period, temp_copies, allowed_mismatch_rate)

        if new_mm > 0:
            if _total_mismatches(text_arr, start, temp_end, consensus, period, n) > max_mm:
                break

        copies = temp_copies
        end = temp_end

    # Extend left
    while start - period >= 0:
        temp_copies = copies + 1
        temp_start = start - period
        new_mm = _hamming_distance(text_arr[temp_start:temp_start + period], consensus, period)
        max_mm = _max_mismatch_threshold(period, temp_copies, allowed_mismatch_rate)

        if new_mm > 0:
            if _total_mismatches(text_arr, temp_start, end, consensus, period, n) > max_mm:
                break

        copies = temp_copies
        start = temp_start

    full_start = start
    full_end = end

    # Partial right extension (exact matching)
    partial_right = 0
    while partial_right < period and full_end + partial_right < n:
        if text_arr[full_end + partial_right] != consensus[partial_right]:
            break
        partial_right += 1
    array_end = full_end + partial_right

    # Partial left extension (exact matching)
    partial_left = 0
    while partial_left < period and full_start - partial_left - 1 >= 0:
        if text_arr[full_start - partial_left - 1] != consensus[period - 1 - partial_left]:
            break
        partial_left += 1
    array_start = full_start - partial_left

    return array_start, array_end, copies, full_start, full_end

cpdef list scan_unit_repeats(const unsigned char[:] text_arr, int n, int unit_len, int min_copies, int max_mismatch):
    """Scan for repeats of a specific unit length."""
    cdef int i = 0
    cdef list results = []
    cdef int count, start_pos, end_pos
    cdef int a_start, a_end, b_start, b_end
    cdef int allowed_errors
    cdef int dist
    cdef bint found_indel
    
    # Calculate dynamic error threshold (15% of unit length or max_mismatch, whichever is higher)
    allowed_errors = max_mismatch
    cdef int dynamic_errors = <int>(unit_len * 0.15)
    if dynamic_errors > allowed_errors:
        allowed_errors = dynamic_errors

    while i + unit_len * min_copies <= n:
        count = 1
        start_pos = i
        
        # Extend right while adjacency holds
        while True:
            a_start = i + (count - 1) * unit_len
            a_end = i + count * unit_len
            b_start = i + count * unit_len
            b_end = b_start + unit_len
            
            if b_end > n:
                break
                
            # 1. Check direct Hamming distance
            dist = _hamming_distance(text_arr[a_start:a_end], text_arr[b_start:b_end], unit_len)
            if dist <= allowed_errors:
                count += 1
                continue
                
            # 2. Check for 1bp Indels
            found_indel = False
            
            # Check shift -1
            if b_start > 0:
                dist = _hamming_distance(text_arr[a_start:a_end], text_arr[b_start-1:b_end-1], unit_len)
                if dist <= allowed_errors:
                    count += 1
                    found_indel = True
            
            if not found_indel and b_end + 1 <= n:
                # Check shift +1
                dist = _hamming_distance(text_arr[a_start:a_end], text_arr[b_start+1:b_end+1], unit_len)
                if dist <= allowed_errors:
                    count += 1
                    found_indel = True
            
            if not found_indel:
                break
        
        if count >= min_copies:
            end_pos = i + count * unit_len
            results.append((i, end_pos))
            i = end_pos
        else:
            i += 1
            
    return results

cpdef list scan_simple_repeats(
    const unsigned char[:] text_arr,
    const unsigned char[:] tier1_mask,
    int n,
    int min_p,
    int max_p,
    int period_step,
    int position_step,
    double allowed_mismatch_rate
):
    """Scan for simple repeats using accelerated logic."""
    cdef int p, i
    cdef int check_len
    cdef int start_pos, end_pos, copies, full_start, full_end
    cdef int array_start, array_end
    cdef list results = []
    cdef int j
    cdef bint match
    
    # Iterate periods
    for p in range(min_p, max_p + 1, period_step):
        i = 0
        check_len = 4
        if p < 4:
            check_len = p
            
        while i < n - p:
            # Skip if masked
            if tier1_mask[i]:
                i += position_step
                continue
                
            # Quick check for periodicity (array_equal replacement)
            if i + p + check_len <= n:
                match = True
                for j in range(check_len):
                    if text_arr[i + j] != text_arr[i + p + j]:
                        match = False
                        break
                
                if match:
                    # Found match, extend
                    # We call the C implementation of extend_with_mismatches directly
                    # Note: extend_with_mismatches returns tuple or None
                    # But we can't call cpdef from cdef easily if we want C speed?
                    # Actually we can call it.
                    # Or better, inline the logic or call a cdef version.
                    # But extend_with_mismatches is cpdef, so it's callable.
                    # However, it returns a Python tuple.
                    # To avoid tuple overhead, we should probably refactor extend_with_mismatches to have a cdef core.
                    # But for now, let's just call it. The extension happens rarely compared to the scan.
                    
                    res = extend_with_mismatches(text_arr, i, p, n, allowed_mismatch_rate)
                    if res is not None:
                        array_start, array_end, copies, full_start, full_end = res
                        
                        # Dynamic copy threshold
                        if (p >= 20 and copies >= 2) or copies >= 3:
                            results.append((full_start, full_end, p))
                            
                            # Skip past this repeat
                            i = full_end
                            continue
            
            i += position_step
            
    return results
