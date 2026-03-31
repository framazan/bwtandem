/*
 * bwt_accel.c — C acceleration for BWT rank queries and backward search.
 *
 * Without Numba, rank queries use numpy's np.count_nonzero(arr[start:end] == code)
 * which allocates temporary boolean + int arrays on every call. This is called
 * ~32 times per k-mer query (2 rank calls × 16 characters). Moving to C
 * eliminates all allocation overhead.
 */
#include <stdlib.h>
#include <string.h>

/*
 * Count occurrences of `code` in arr[start..end-1].
 * Direct replacement for _count_equal_range.
 */
int count_equal_range(
    const unsigned char *arr,
    int start,
    int end,
    int code
)
{
    int count = 0;
    /* Unroll for performance */
    int i = start;
    unsigned char c = (unsigned char)code;
    for (; i + 7 < end; i += 8) {
        count += (arr[i]   == c);
        count += (arr[i+1] == c);
        count += (arr[i+2] == c);
        count += (arr[i+3] == c);
        count += (arr[i+4] == c);
        count += (arr[i+5] == c);
        count += (arr[i+6] == c);
        count += (arr[i+7] == c);
    }
    for (; i < end; i++) {
        count += (arr[i] == c);
    }
    return count;
}

/*
 * Rank query: count occurrences of `code` in bwt[0..pos-1] using checkpoints.
 *
 * checkpoints: array where checkpoints[i] = count of code in bwt[0..i*sample_rate-1]
 *              checkpoints[0] = 0
 * bwt: BWT array
 * pos: query position (count in bwt[0..pos-1])
 * sample_rate: checkpoint sampling rate
 */
int rank_query(
    const unsigned char *bwt,
    const int *checkpoints,
    int n_checkpoints,
    int pos,
    int n,
    int code,
    int sample_rate
)
{
    if (pos <= 0) return 0;
    if (pos > n) pos = n;

    int cp_idx = pos / sample_rate;
    if (cp_idx >= n_checkpoints) cp_idx = n_checkpoints - 1;
    int base = checkpoints[cp_idx];
    int cp_pos = cp_idx * sample_rate;

    if (pos > cp_pos) {
        base += count_equal_range(bwt, cp_pos, pos, code);
    }
    return base;
}

/*
 * Backward search: find SA interval [sp, ep] for pattern in BWT.
 *
 * char_counts: C[c] = number of characters < c in text
 * char_totals: total[c] = number of occurrences of c in text
 * checkpoints_flat: flattened checkpoint arrays, one per character code
 * cp_offsets: cp_offsets[code] = start offset into checkpoints_flat for this code
 * cp_lengths: cp_lengths[code] = number of checkpoints for this code
 *
 * Returns: sp and ep via output pointers. Returns 1 if found, 0 if not.
 */
int backward_search(
    const unsigned char *bwt,
    int n,
    const unsigned char *pattern,
    int pat_len,
    const int *char_counts,     /* size 256: C[c] */
    const int *char_totals,     /* size 256: total[c] */
    const int *checkpoints_flat,
    const int *cp_offsets,      /* size 256 */
    const int *cp_lengths,      /* size 256 */
    int sample_rate,
    int *out_sp,
    int *out_ep
)
{
    if (pat_len == 0) {
        *out_sp = 0;
        *out_ep = n - 1;
        return 1;
    }

    unsigned char c = pattern[pat_len - 1];
    if (char_totals[c] == 0) {
        *out_sp = -1; *out_ep = -1;
        return 0;
    }

    int sp = char_counts[c];
    int ep = sp + char_totals[c] - 1;

    for (int i = pat_len - 2; i >= 0; i--) {
        c = pattern[i];
        if (char_totals[c] == 0) {
            *out_sp = -1; *out_ep = -1;
            return 0;
        }

        const int *cp = checkpoints_flat + cp_offsets[c];
        int n_cp = cp_lengths[c];

        int new_sp = char_counts[c] + rank_query(bwt, cp, n_cp, sp, n, c, sample_rate);
        int new_ep = char_counts[c] + rank_query(bwt, cp, n_cp, ep + 1, n, c, sample_rate) - 1;

        if (new_sp > new_ep) {
            *out_sp = -1; *out_ep = -1;
            return 0;
        }
        sp = new_sp;
        ep = new_ep;
    }

    *out_sp = sp;
    *out_ep = ep;
    return 1;
}

/*
 * Kasai's algorithm for LCP array construction in O(n).
 * text_codes: sequence as byte array
 * sa: suffix array (int32)
 * n: length
 * lcp_out: output LCP array (int32, pre-allocated, size n)
 */
void kasai_lcp(
    const unsigned char *text_codes,
    const int *sa,
    int n,
    int *lcp_out
)
{
    if (n <= 0) return;
    int *rank = (int *)malloc(n * sizeof(int));
    if (!rank) return;
    for (int i = 0; i < n; i++) {
        rank[sa[i]] = i;
    }

    memset(lcp_out, 0, n * sizeof(int));
    int h = 0;
    for (int i = 0; i < n; i++) {
        int r = rank[i];
        if (r > 0) {
            int j = sa[r - 1];
            while (i + h < n && j + h < n && text_codes[i + h] == text_codes[j + h]) {
                h++;
            }
            lcp_out[r] = h;
            if (h > 0) h--;
        } else {
            h = 0;
        }
    }
    free(rank);
}

/*
 * Batch backward search: search multiple patterns at once.
 * Avoids per-call Python→C overhead.
 *
 * patterns: concatenated pattern bytes
 * pat_offsets: start offset of each pattern in `patterns`
 * pat_lengths: length of each pattern
 * n_patterns: number of patterns
 * out_sps, out_eps: output arrays of size n_patterns
 */
void batch_backward_search(
    const unsigned char *bwt,
    int n,
    const unsigned char *patterns,
    const int *pat_offsets,
    const int *pat_lengths,
    int n_patterns,
    const int *char_counts,
    const int *char_totals,
    const int *checkpoints_flat,
    const int *cp_offsets,
    const int *cp_lengths,
    int sample_rate,
    int *out_sps,
    int *out_eps
)
{
    for (int q = 0; q < n_patterns; q++) {
        backward_search(
            bwt, n,
            patterns + pat_offsets[q], pat_lengths[q],
            char_counts, char_totals,
            checkpoints_flat, cp_offsets, cp_lengths,
            sample_rate,
            &out_sps[q], &out_eps[q]
        );
    }
}
