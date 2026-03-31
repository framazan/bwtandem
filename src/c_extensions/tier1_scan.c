#include <stdlib.h>
#include <string.h>

/*
 * Fast period-k run detection for Tier 1 STR finding.
 *
 * Compares text[i] with text[i+k] to find runs of consecutive matches,
 * then returns candidate arrays as (start, end, copies) triples.
 *
 * Returns: number of candidates found.
 * Output is written to out_starts, out_ends, out_copies arrays
 * (caller must allocate with max_candidates capacity).
 */
int find_period_runs(
    const unsigned char *text,  /* input sequence as byte array */
    int n,                      /* sequence length */
    int k,                      /* period (motif length) */
    int min_seed_copies,        /* minimum copies for a seed */
    const unsigned char *seen,  /* boolean mask of already-found positions */
    int *out_starts,            /* output: candidate start positions */
    int *out_ends,              /* output: candidate end positions */
    int *out_copies,            /* output: candidate copy counts */
    int max_candidates          /* max output capacity */
)
{
    int count = 0;
    int limit = n - k;
    int i = 0;

    while (i < limit && count < max_candidates) {
        /* Skip non-matching positions */
        if (text[i] != text[i + k]) {
            i++;
            continue;
        }

        /* Found start of a matching run */
        int run_start = i;
        int j = i + 1;
        while (j < limit && text[j] == text[j + k]) {
            j++;
        }

        /* Array spans from run_start to j + k */
        int array_start = run_start;
        int array_end = j + k;
        int seed_length = array_end - array_start;
        int seed_copies = seed_length / k;

        i = j;  /* advance past run */

        if (seed_copies < min_seed_copies)
            continue;

        /* Skip if start or midpoint already masked */
        int midpoint = (array_start + array_end) / 2;
        if (midpoint >= n) midpoint = n - 1;
        if (seen[array_start] || seen[midpoint])
            continue;

        /* Skip sentinel ($=36) and N (78,110) in motif */
        int bad_motif = 0;
        int mi;
        for (mi = 0; mi < k; mi++) {
            unsigned char ch = text[array_start + mi];
            if (ch == 36 || ch == 78 || ch == 110) {
                bad_motif = 1;
                break;
            }
        }
        if (bad_motif)
            continue;

        out_starts[count] = array_start;
        out_ends[count] = array_end;
        out_copies[count] = seed_copies;
        count++;
    }

    return count;
}
