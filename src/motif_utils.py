import numpy as np
from typing import List, Tuple, Dict, Iterator, Optional
from collections import Counter
import math
from .models import AlignmentResult, RepeatAlignmentSummary

class MotifUtils:
    """Utilities for canonical motif handling."""

    @staticmethod
    def get_canonical_motif(motif: str) -> str:
        """Get lexicographically smallest rotation of motif."""
        if not motif:
            return motif

        rotations = [motif[i:] + motif[:i] for i in range(len(motif))]
        return min(rotations)

    @staticmethod
    def reverse_complement(seq: str) -> str:
        """Get reverse complement of DNA sequence."""
        complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        return ''.join(complement_map.get(b, b) for b in reversed(seq))

    @staticmethod
    def get_canonical_motif_stranded(motif: str) -> Tuple[str, str]:
        """Get canonical motif considering both strands.

        Returns:
            (canonical_motif, strand) where strand is '+' or '-'
        """
        if not motif:
            return motif, '+'

        # Get all rotations of forward strand
        forward_rotations = [motif[i:] + motif[:i] for i in range(len(motif))]
        forward_canonical = min(forward_rotations)

        # Get all rotations of reverse complement
        rc = MotifUtils.reverse_complement(motif)
        rc_rotations = [rc[i:] + rc[:i] for i in range(len(rc))]
        rc_canonical = min(rc_rotations)

        # Return lexicographically smallest
        if forward_canonical <= rc_canonical:
            return forward_canonical, '+'
        else:
            return rc_canonical, '-'

    @staticmethod
    def is_primitive_motif(motif: str) -> bool:
        """Check if motif is not a repetition of a shorter motif."""
        n = len(motif)
        for i in range(1, n):
            if n % i == 0:
                period = motif[:i]
                if period * (n // i) == motif:
                    return False
        return True

    @staticmethod
    def calculate_entropy(seq: str) -> float:
        """Calculate Shannon entropy of sequence (bits per base)."""
        if not seq:
            return 0.0

        from collections import Counter
        counts = Counter(seq)
        n = len(seq)
        entropy = 0.0

        for count in counts.values():
            if count > 0:
                p = count / n
                entropy -= p * np.log2(p)

        return entropy

    @staticmethod
    def is_transition(base1: str, base2: str) -> bool:
        """Check if a base change is a transition (A↔G or C↔T).

        Transitions are more common than transversions in biology.
        Purines: A, G (transitions within purines: A↔G)
        Pyrimidines: C, T (transitions within pyrimidines: C↔T)
        """
        if base1 == base2:
            return True  # No change

        transitions = {
            ('A', 'G'), ('G', 'A'),  # Purine transitions
            ('C', 'T'), ('T', 'C')   # Pyrimidine transitions
        }
        return (base1, base2) in transitions

    @staticmethod
    def hamming_distance(s1: str, s2: str) -> int:
        """Calculate Hamming distance between two strings of equal length."""
        if len(s1) != len(s2):
            return max(len(s1), len(s2))

        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    @staticmethod
    def hamming_distance_array(arr1: np.ndarray, arr2: np.ndarray) -> int:
        """Calculate Hamming distance between two uint8 arrays."""
        if arr1.size != arr2.size:
            return max(arr1.size, arr2.size)

        return int(np.count_nonzero(arr1 != arr2))

    @staticmethod
    def count_transversions_array(arr1: np.ndarray, arr2: np.ndarray) -> int:
        """Count transversions (non-transition mismatches) between two uint8 arrays.

        Returns number of transversion changes (A↔C, A↔T, G↔C, G↔T).
        """
        if arr1.size != arr2.size:
            return max(arr1.size, arr2.size)

        # ASCII codes: A=65, C=67, G=71, T=84
        transversions = 0
        for i in range(arr1.size):
            b1, b2 = arr1[i], arr2[i]
            if b1 != b2:
                # Convert to characters for transition check
                c1 = chr(b1) if 65 <= b1 <= 84 else 'N'
                c2 = chr(b2) if 65 <= b2 <= 84 else 'N'
                if not MotifUtils.is_transition(c1, c2):
                    transversions += 1

        return transversions

    @staticmethod
    def edit_distance(a: str, b: str) -> int:
        """Compute Levenshtein edit distance between two short strings."""
        la, lb = len(a), len(b)
        if la == 0:
            return lb
        if lb == 0:
            return la

        prev = list(range(lb + 1))
        curr = [0] * (lb + 1)

        for i in range(1, la + 1):
            curr[0] = i
            ai = a[i - 1]
            for j in range(1, lb + 1):
                cost = 0 if ai == b[j - 1] else 1
                curr[j] = min(
                    prev[j] + 1,      # deletion
                    curr[j - 1] + 1,  # insertion
                    prev[j - 1] + cost  # substitution
                )
            prev, curr = curr, prev

        return prev[lb]

    @staticmethod
    def _align_unit_to_window(motif: str, window: str, max_indel: int,
                              mismatch_tolerance: int) -> Optional[AlignmentResult]:
        """Align motif to a window allowing mismatches and small indels."""
        m = len(motif)
        n = len(window)

        if m == 0 or n == 0:
            return None

        max_indel = max(0, max_indel)
        mismatch_tolerance = max(0, mismatch_tolerance)

        lower = max(0, m - max_indel)
        upper = min(n, m + max_indel)
        if lower > upper:
            return None

        inf = m + n + 10
        dp = [[inf] * (n + 1) for _ in range(m + 1)]
        ptr = [[''] * (n + 1) for _ in range(m + 1)]

        dp[0][0] = 0
        for j in range(1, n + 1):
            dp[0][j] = j
            ptr[0][j] = 'I'
        for i in range(1, m + 1):
            dp[i][0] = i
            ptr[i][0] = 'D'

        band_extra = max_indel + 2

        for i in range(1, m + 1):
            j_min = max(1, i - band_extra)
            j_max = min(n, i + band_extra)
            for j in range(j_min, j_max + 1):
                sub_cost = dp[i - 1][j - 1] + (motif[i - 1] != window[j - 1])
                del_cost = dp[i - 1][j] + 1
                ins_cost = dp[i][j - 1] + 1

                best_cost = sub_cost
                best_ptr = 'M' if motif[i - 1] == window[j - 1] else 'S'

                if del_cost < best_cost:
                    best_cost = del_cost
                    best_ptr = 'D'
                if ins_cost < best_cost:
                    best_cost = ins_cost
                    best_ptr = 'I'

                dp[i][j] = best_cost
                ptr[i][j] = best_ptr

        best_j = -1
        best_cost = inf
        for j in range(lower, upper + 1):
            cost = dp[m][j]
            if cost < best_cost:
                best_cost = cost
                best_j = j

        if best_j <= 0 or best_cost >= inf:
            return None

        aligned_ref = []
        aligned_query = []
        i, j = m, best_j
        while i > 0 or j > 0:
            op = ptr[i][j]
            if op in ('M', 'S'):
                aligned_ref.append(motif[i - 1])
                aligned_query.append(window[j - 1])
                i -= 1
                j -= 1
            elif op == 'D':
                aligned_ref.append(motif[i - 1])
                aligned_query.append('-')
                i -= 1
            elif op == 'I':
                aligned_ref.append('-')
                aligned_query.append(window[j - 1])
                j -= 1
            else:  # Should only occur at origin
                break

        aligned_ref.reverse()
        aligned_query.reverse()

        operations: List[Tuple] = []
        observed_bases: List[Tuple[int, str]] = []
        mismatch_count = 0
        insertion_len = 0
        deletion_len = 0

        ref_pos = 0
        pending_ins: List[str] = []
        pending_ins_pos = 0
        pending_del_len = 0
        pending_del_pos = 0

        for r, q in zip(aligned_ref, aligned_query):
            if r == '-':
                if not pending_ins:
                    pending_ins_pos = ref_pos
                pending_ins.append(q)
                continue

            if pending_ins:
                ins_seq = ''.join(pending_ins)
                operations.append(('ins', pending_ins_pos, ins_seq))
                insertion_len += len(ins_seq)
                pending_ins = []
                pending_ins_pos = 0

            ref_pos += 1

            if q == '-':
                if pending_del_len == 0:
                    pending_del_pos = ref_pos
                pending_del_len += 1
                continue

            if pending_del_len:
                operations.append(('del', pending_del_pos, pending_del_len))
                deletion_len += pending_del_len
                pending_del_len = 0

            observed_bases.append((ref_pos - 1, q))
            if r != q:
                operations.append(('sub', ref_pos, r, q))
                mismatch_count += 1

        if pending_ins:
            ins_seq = ''.join(pending_ins)
            operations.append(('ins', pending_ins_pos, ins_seq))
            insertion_len += len(ins_seq)

        if pending_del_len:
            operations.append(('del', pending_del_pos, pending_del_len))
            deletion_len += pending_del_len

        if mismatch_count > mismatch_tolerance:
            return None
        if insertion_len > max_indel or deletion_len > max_indel:
            return None

        return AlignmentResult(
            consumed=best_j,
            unit_sequence=window[:best_j],
            mismatch_count=mismatch_count,
            insertion_length=insertion_len,
            deletion_length=deletion_len,
            operations=operations,
            observed_bases=observed_bases,
            edit_distance=best_cost
        )

    @staticmethod
    def _consensus_from_counts(counts: List[Counter], fallback: str) -> str:
        """Build consensus string from per-position base counts."""
        consensus = []
        for idx, counter in enumerate(counts):
            if counter:
                base, _ = counter.most_common(1)[0]
                consensus.append(base)
            else:
                consensus.append(fallback[idx] if idx < len(fallback) else 'N')
        return ''.join(consensus)

    @staticmethod
    def align_repeat_region(sequence: str, start: int, end: int, motif_template: str,
                            mismatch_fraction: float = 0.1,
                            max_indel: Optional[int] = None,
                            min_copies: int = 3) -> Optional[RepeatAlignmentSummary]:
        """Align sequential copies of a motif within a sequence region."""
        if not motif_template:
            return None

        seq_len = len(sequence)
        if seq_len == 0:
            return None

        start = max(0, start)
        end = min(seq_len, end if end > start else seq_len)

        motif_len = len(motif_template)
        if motif_len == 0:
            return None

        tolerance = max(1, int(math.floor(motif_len * mismatch_fraction)))
        if max_indel is None:
            max_indel = max(1, min(10, motif_len // 2 if motif_len >= 4 else 1))
        else:
            max_indel = max(0, max_indel)

        position_counts: List[Counter] = [Counter() for _ in range(motif_len)]
        copy_sequences: List[str] = []
        operations_by_copy: List[List[Tuple]] = []
        error_counts: List[int] = []

        total_insertions = 0
        total_deletions = 0

        current_motif = motif_template
        pos = start
        safety_limit = min(seq_len, max(end, start + motif_len * min_copies) + max(motif_len * 3, max_indel * 4))

        while pos < safety_limit:
            window_end = min(seq_len, pos + motif_len + max_indel)
            window = sequence[pos:window_end]
            if len(window) < motif_len - max_indel:
                break

            result = MotifUtils._align_unit_to_window(current_motif, window, max_indel, tolerance)
            if result is None or result.consumed == 0:
                print(f"DEBUG: Alignment failed at pos {pos}. Window: {window}, Motif: {current_motif}")
                break

            copy_sequences.append(result.unit_sequence)
            operations_by_copy.append(result.operations)
            error_counts.append(result.error_count)
            total_insertions += result.insertion_length
            total_deletions += result.deletion_length

            for motif_idx, base in result.observed_bases:
                if 0 <= motif_idx < motif_len:
                    position_counts[motif_idx][base] += 1

            pos += result.consumed
            current_motif = MotifUtils._consensus_from_counts(position_counts, current_motif)

        copies = len(copy_sequences)
        if copies < min_copies:
            return None

        consumed_len = pos - start
        if consumed_len <= 0:
            return None

        consensus = MotifUtils._consensus_from_counts(position_counts, current_motif)
        total_errors = sum(error_counts)
        denom = copies * motif_len
        mismatch_rate = total_errors / denom if denom > 0 else 0.0
        max_errors_per_copy = max(error_counts) if error_counts else 0

        variations: List[str] = []
        for idx, ops in enumerate(operations_by_copy, 1):
            for op in ops:
                if not op:
                    continue
                kind = op[0]
                if kind == 'sub':
                    _, pos_idx, ref_base, alt_base = op
                    variations.append(f"{idx}:{pos_idx}:{ref_base}>{alt_base}")
                elif kind == 'ins':
                    _, pos_idx, inserted = op
                    if inserted:
                        variations.append(f"{idx}:{pos_idx}:ins({inserted})")
                elif kind == 'del':
                    _, pos_idx, length = op
                    if length > 0:
                        variations.append(f"{idx}:{pos_idx}:del({length})")

        return RepeatAlignmentSummary(
            consensus=consensus,
            motif_len=motif_len,
            copies=copies,
            consumed_length=consumed_len,
            mismatch_rate=mismatch_rate,
            max_errors_per_copy=max_errors_per_copy,
            variations=variations,
            copy_sequences=copy_sequences,
            total_insertions=total_insertions,
            total_deletions=total_deletions,
            error_counts=error_counts
        )

    @staticmethod
    def is_insertion_variant(candidate: str, consensus: str) -> bool:
        """Return True if removing a single base from candidate yields consensus."""
        if len(candidate) != len(consensus) + 1:
            return False
        for i in range(len(candidate)):
            if candidate[:i] + candidate[i + 1:] == consensus:
                return True
        return False

    @staticmethod
    def is_deletion_variant(candidate: str, consensus: str) -> bool:
        """Return True if inserting a single base into candidate yields consensus."""
        if len(candidate) + 1 != len(consensus):
            return False
        for i in range(len(consensus)):
            if consensus[:i] + consensus[i + 1:] == candidate:
                return True
        return False

    @staticmethod
    def smallest_period_str(s: str) -> int:
        """Return length of the smallest period of string s."""
        if not s:
            return 0
        n = len(s)
        for p in range(1, n + 1):
            if n % p == 0 and s == s[:p] * (n // p):
                return p
        return n

    @staticmethod
    def normalize_variant(candidate: str, consensus: str) -> str:
        """Rotate variant to best align with consensus (minimal edit distance)."""
        if not candidate:
            return candidate

        best = candidate
        best_cost = MotifUtils.edit_distance(candidate, consensus)

        if len(candidate) == 1:
            return candidate

        for shift in range(1, len(candidate)):
            rotated = candidate[shift:] + candidate[:shift]
            cost = MotifUtils.edit_distance(rotated, consensus)
            if cost < best_cost or (cost == best_cost and rotated < best):
                best = rotated
                best_cost = cost

        return best

    @staticmethod
    def rotate_deletion_variant(candidate: str, consensus: str) -> str:
        """Rotate shorter variant so it no longer begins with the consensus prefix."""
        if not candidate:
            return candidate

        rotated = candidate
        for _ in range(len(candidate)):
            if rotated[0] != consensus[0]:
                return rotated
            rotated = rotated[1:] + rotated[:1]
        return rotated

    @staticmethod
    def build_consensus_motif(sequences: List[str]) -> Tuple[str, float]:
        """Build consensus motif from multiple aligned sequences using majority vote.

        Returns:
            (consensus, avg_mismatch_rate) - consensus sequence and average mismatch rate
        """
        if not sequences:
            return "", 0.0

        if len(sequences) == 1:
            return sequences[0], 0.0

        motif_len = len(sequences[0])
        consensus = []
        total_mismatches = 0

        for pos in range(motif_len):
            bases = [seq[pos] for seq in sequences if pos < len(seq)]
            if not bases:
                consensus.append('N')
                continue

            # Majority vote
            from collections import Counter
            counts = Counter(bases)
            most_common = counts.most_common(1)[0][0]
            consensus.append(most_common)

            # Count mismatches at this position
            mismatches = len(bases) - counts[most_common]
            total_mismatches += mismatches

        total_bases = len(sequences) * motif_len
        avg_mismatch_rate = total_mismatches / total_bases if total_bases > 0 else 0.0

        return ''.join(consensus), avg_mismatch_rate

    @staticmethod
    def build_consensus_motif_array(text_arr: np.ndarray, start: int, motif_len: int,
                                   n_copies: int) -> Tuple[np.ndarray, float, int]:
        """Build consensus motif from array copies using majority vote.

        Returns:
            (consensus_array, mismatch_rate, max_mismatches_per_copy)
        """
        if n_copies == 0 or motif_len == 0:
            return np.array([], dtype=np.uint8), 0.0, 0

        consensus = np.zeros(motif_len, dtype=np.uint8)
        total_mismatches = 0
        max_mismatches_per_copy = 0

        # Collect all copies
        copies = []
        for i in range(n_copies):
            copy_start = start + i * motif_len
            copy_end = copy_start + motif_len
            if copy_end > text_arr.size:
                break
            copy_arr = text_arr[copy_start:copy_end]
            copies.append(copy_arr)

        if not copies:
            return np.array([], dtype=np.uint8), 0.0, 0

        # Build consensus by majority vote at each position
        for pos in range(motif_len):
            bases = [copy[pos] for copy in copies if pos < len(copy)]
            if not bases:
                consensus[pos] = ord('N')
                continue

            # Find most common base
            unique, counts = np.unique(bases, return_counts=True)
            most_common_idx = np.argmax(counts)
            consensus[pos] = unique[most_common_idx]

        # Calculate mismatch statistics
        for copy in copies:
            mismatches = MotifUtils.hamming_distance_array(copy, consensus)
            total_mismatches += mismatches
            max_mismatches_per_copy = max(max_mismatches_per_copy, mismatches)

        total_bases = len(copies) * motif_len
        mismatch_rate = total_mismatches / total_bases if total_bases > 0 else 0.0

        return consensus, mismatch_rate, max_mismatches_per_copy

    @staticmethod
    def summarize_variations_array(text_arr: np.ndarray, start: int, end: int, motif_len: int,
                                   consensus_arr: np.ndarray) -> List[str]:
        """Summarize per-copy variations relative to consensus, allowing small indels."""
        if text_arr.size == 0 or motif_len <= 0:
            return []

        sequence = text_arr.tobytes().decode('ascii', errors='replace')
        start = max(0, start)
        end = min(len(sequence), end if end > start else len(sequence))

        if end <= start:
            return []

        if consensus_arr.size > 0:
            motif_template = consensus_arr.tobytes().decode('ascii', errors='replace')
        else:
            motif_template = sequence[start:start + motif_len]

        summary = MotifUtils.align_repeat_region(
            sequence,
            start,
            end,
            motif_template,
            mismatch_fraction=0.3,  # Allow ~30% mismatches to capture variations
            min_copies=1
        )
        if not summary:
            return []
        return summary.variations

    @staticmethod
    def calculate_composition(sequence: str) -> Dict[str, float]:
        """Calculate nucleotide composition as percentages.

        Returns:
            Dictionary with A, C, G, T percentages
        """
        if not sequence:
            return {'A': 0.0, 'C': 0.0, 'G': 0.0, 'T': 0.0}

        from collections import Counter
        counts = Counter(sequence.upper())
        total = len(sequence)

        composition = {
            'A': (counts.get('A', 0) / total) * 100.0,
            'C': (counts.get('C', 0) / total) * 100.0,
            'G': (counts.get('G', 0) / total) * 100.0,
            'T': (counts.get('T', 0) / total) * 100.0,
        }

        return composition

    @staticmethod
    def calculate_trf_score(consensus: str, copies: int, mismatch_rate: float, length: int) -> int:
        """Calculate TRF-style alignment score.

        TRF uses match/mismatch/indel scoring. We approximate:
        - Match: +2 points
        - Mismatch: -7 points
        - Indel: -7 points (we use 0 indels for Hamming distance)

        Returns:
            Alignment score (integer)
        """
        total_bases = length
        matches = total_bases * (1.0 - mismatch_rate)
        mismatches = total_bases * mismatch_rate

        # TRF scoring parameters (approximately)
        match_score = 2
        mismatch_penalty = 7

        score = int((matches * match_score) - (mismatches * mismatch_penalty))
        return max(0, score)  # Don't allow negative scores

    @staticmethod
    def calculate_trf_statistics(text_arr: np.ndarray, start: int, end: int,
                                 consensus_motif: str, copies: int,
                                 mismatch_rate: float) -> Tuple[float, float, int, Dict[str, float], float, str]:
        """Calculate TRF-compatible statistics for a repeat.

        Returns:
            (percent_matches, percent_indels, score, composition, entropy, actual_sequence)
        """
        # Extract actual sequence
        if end <= text_arr.size:
            actual_sequence = text_arr[start:end].tobytes().decode('ascii', errors='replace')
        else:
            actual_sequence = consensus_motif * int(copies)

        # Percent matches (inverse of mismatch rate)
        percent_matches = (1.0 - mismatch_rate) * 100.0

        # Percent indels (we use Hamming distance, so 0 indels)
        percent_indels = 0.0

        # Composition
        composition = MotifUtils.calculate_composition(consensus_motif)

        # Entropy (already calculated, but recalculate for consistency)
        entropy = MotifUtils.calculate_entropy(consensus_motif)

        # Score
        length = end - start
        score = MotifUtils.calculate_trf_score(consensus_motif, copies, mismatch_rate, length)

        return percent_matches, percent_indels, score, composition, entropy, actual_sequence

    @staticmethod
    def enumerate_motifs(k: int, alphabet: str = "ACGT") -> Iterator[str]:
        """Generate all canonical primitive motifs of length k."""
        def generate_strings(length, current=""):
            if length == 0:
                canonical = MotifUtils.get_canonical_motif(current)
                if canonical == current and MotifUtils.is_primitive_motif(current):
                    yield current
                return

            for char in alphabet:
                yield from generate_strings(length - 1, current + char)

        yield from generate_strings(k)
