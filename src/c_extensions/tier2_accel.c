#include <stdlib.h>
#include <string.h>

/*
 * Fast smallest_period_str: find the smallest period p such that
 * motif[0..n-1] == motif[0..p-1] repeated n/p times.
 * Returns p (the primitive period length).
 */
int smallest_period_str(
    const unsigned char *motif,
    int n
)
{
    for (int p = 1; p <= n; p++) {
        if (n % p != 0) continue;
        int valid = 1;
        for (int i = p; i < n; i++) {
            if (motif[i] != motif[i % p]) {
                valid = 0;
                break;
            }
        }
        if (valid) return p;
    }
    return n;
}

/*
 * Fast smallest_period_str_approx: find approximate primitive period
 * with up to max_error_rate mismatch tolerance.
 * Returns the smallest period where mismatch rate <= max_error_rate.
 */
int smallest_period_str_approx(
    const unsigned char *motif,
    int n,
    double max_error_rate
)
{
    for (int p = 1; p <= n / 2; p++) {
        double repeats = (double)n / p;
        if (repeats < 2.0) continue;

        int mismatches = 0;
        for (int i = 0; i < n; i++) {
            if (motif[i] != motif[i % p]) {
                mismatches++;
            }
        }
        double rate = (double)mismatches / n;
        if (rate <= max_error_rate) return p;
    }
    return n;
}

/*
 * Fast hamming distance between two byte arrays.
 */
int hamming_distance(
    const unsigned char *a,
    const unsigned char *b,
    int n
)
{
    int dist = 0;
    /* Process 8 bytes at a time for speed */
    int i = 0;
    for (; i + 7 < n; i += 8) {
        dist += (a[i]   != b[i]);
        dist += (a[i+1] != b[i+1]);
        dist += (a[i+2] != b[i+2]);
        dist += (a[i+3] != b[i+3]);
        dist += (a[i+4] != b[i+4]);
        dist += (a[i+5] != b[i+5]);
        dist += (a[i+6] != b[i+6]);
        dist += (a[i+7] != b[i+7]);
    }
    for (; i < n; i++) {
        dist += (a[i] != b[i]);
    }
    return dist;
}

/*
 * Fast mismatch-tolerant extension from a seed position.
 * Extends both forward and backward from seed_pos using period-sized windows.
 *
 * Returns: number of copies found.
 * out_start and out_end are set to the full extended region boundaries.
 */
int extend_mismatch_c(
    const unsigned char *text,
    int n,
    int seed_pos,
    int period,
    double max_mismatch_rate,
    int *out_start,
    int *out_end
)
{
    if (seed_pos < 0 || seed_pos + period > n || period <= 0)
        return 0;

    const unsigned char *motif = text + seed_pos;
    int max_mm = (int)(period * max_mismatch_rate + 0.5);
    if (max_mm < 1) max_mm = 1;

    int start = seed_pos;
    int end = seed_pos + period;

    /* Extend forward */
    int pos = seed_pos + period;
    while (pos + period <= n) {
        int mm = 0;
        for (int i = 0; i < period; i++) {
            if (text[pos + i] != motif[i]) {
                mm++;
                if (mm > max_mm) break;
            }
        }
        if (mm <= max_mm) {
            end = pos + period;
            pos += period;
        } else {
            break;
        }
    }

    /* Extend backward */
    pos = seed_pos - period;
    while (pos >= 0) {
        int mm = 0;
        for (int i = 0; i < period; i++) {
            if (text[pos + i] != motif[i]) {
                mm++;
                if (mm > max_mm) break;
            }
        }
        if (mm <= max_mm) {
            start = pos;
            pos -= period;
        } else {
            break;
        }
    }

    *out_start = start;
    *out_end = end;
    return (end - start) / period;
}

/*
 * Batch process LCP candidates: for each (period, seed_pos), extend with
 * mismatches and compute primitive period. Skip covered positions.
 *
 * Returns number of valid results.
 * Results are written to out arrays.
 */
int batch_process_lcp_candidates(
    const unsigned char *text,
    int n,
    const int *periods,        /* candidate periods */
    const int *seed_positions,  /* candidate seed positions */
    int n_candidates,
    double max_mismatch_rate,
    int min_copies,
    unsigned char *covered_mask,  /* boolean mask, updated in-place */
    int *out_starts,
    int *out_ends,
    int *out_periods,          /* primitive period for each result */
    int *out_copies,
    int max_results
)
{
    int count = 0;

    for (int ci = 0; ci < n_candidates && count < max_results; ci++) {
        int period = periods[ci];
        int seed_pos = seed_positions[ci];

        if (seed_pos < 0 || seed_pos + period > n)
            continue;
        if (covered_mask[seed_pos])
            continue;

        /* Extend with mismatches */
        int ext_start, ext_end;
        int copies = extend_mismatch_c(text, n, seed_pos, period, max_mismatch_rate,
                                        &ext_start, &ext_end);

        if (copies < min_copies)
            continue;

        /* Compute primitive period */
        int prim_period = smallest_period_str(text + ext_start, period);
        if (prim_period == period) {
            prim_period = smallest_period_str_approx(text + ext_start, period, 0.02);
        }

        out_starts[count] = ext_start;
        out_ends[count] = ext_end;
        out_periods[count] = prim_period;
        out_copies[count] = copies;
        count++;

        /* Mark covered region */
        int mark_end = ext_end < n ? ext_end : n;
        for (int i = ext_start; i < mark_end; i++) {
            covered_mask[i] = 1;
        }
    }

    return count;
}
