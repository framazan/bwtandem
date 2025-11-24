import numpy as np
from typing import List, Tuple, Dict, Union

# Optional: JIT acceleration with numba when available
HAVE_NUMBA = False
try:
    import numba as _nb  # type: ignore
    HAVE_NUMBA = True
except Exception:
    _nb = None  # type: ignore

if HAVE_NUMBA:
    @_nb.njit(cache=True)
    def _count_equal_range(arr: np.ndarray, start: int, end: int, code: int) -> int:  # type: ignore
        c = 0
        for i in range(start, end):
            if arr[i] == code:
                c += 1
        return c

    @_nb.njit(cache=True)
    def _kasai_lcp_uint8(text_codes: np.ndarray, sa: np.ndarray) -> np.ndarray:  # type: ignore
        n = text_codes.size
        lcp = np.zeros(n, dtype=np.int32)
        rank = np.zeros(n, dtype=np.int32)
        for i in range(n):
            rank[sa[i]] = i
        h = 0
        for i in range(n):
            r = rank[i]
            if r > 0:
                j = sa[r - 1]
                while i + h < n and j + h < n and text_codes[i + h] == text_codes[j + h]:
                    h += 1
                lcp[r] = h
                if h > 0:
                    h -= 1
        return lcp
else:
    # Dummy decorator to avoid syntax errors if someone tries to use it
    def _dummy_jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def _count_equal_range(arr: np.ndarray, start: int, end: int, code: int) -> int:
        # Pure-python/numpy fallback
        return int(np.count_nonzero(arr[start:end] == code))

    def _kasai_lcp_uint8(text_codes: np.ndarray, sa: np.ndarray) -> np.ndarray:
        # Fallback non-jitted Kasai
        n = text_codes.size
        lcp = np.zeros(n, dtype=np.int32)
        rank = np.zeros(n, dtype=np.int32)
        for i in range(n):
            rank[sa[i]] = i
        h = 0
        for i in range(n):
            r = rank[i]
            if r > 0:
                j = sa[r - 1]
                while i + h < n and j + h < n and text_codes[i + h] == text_codes[j + h]:
                    h += 1
                lcp[r] = h
                if h > 0:
                    h -= 1
        return lcp


class BWTCore:
    """Core BWT construction and FM-index operations.
    """

    # Base encoding for bit-masking (bcftools-inspired)
    BASE_TO_BITS = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}  # 2 bits per base
    BITS_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    def __init__(self, text: str, sa_sample_rate: int = 32, occ_sample_rate: int = 128):
        """
        Initialize BWT with FM-index.

        Args:
            text: Input text (should end with a single '$' sentinel not present elsewhere)
            sa_sample_rate: Sample every nth suffix array position for space efficiency
            occ_sample_rate: Occurrence checkpoints every nth position to reduce memory
        """
        self.text: str = text
        self.n = len(text)
        self.sa_sample_rate = sa_sample_rate
        self.occ_sample_rate = occ_sample_rate


        self.text_arr = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)

        # Build k-mer hash for fast lookups (performance optimization)
        self._build_kmer_hash()

        # Build suffix array and BWT (memory-efficient)
        self.suffix_array = self._build_suffix_array()
        self.bwt_arr = self._build_bwt_array()
        self.alphabet = sorted(set(text))
        self.char_to_code = {c: ord(c) for c in self.alphabet}
        self.code_to_char = {ord(c): c for c in self.alphabet}
        self.char_counts, self.char_totals = self._build_char_counts()
        self.char_counts_code = {ord(k): v for k, v in self.char_counts.items()}
        self.char_totals_code = {ord(k): v for k, v in self.char_totals.items()}
        self.occ_checkpoints = self._build_occurrence_checkpoints()
        self.sampled_sa = self._sample_suffix_array()

    def _build_kmer_hash(self, k: int = 8):
        """Build hash table for k-mer positions (bcftools-inspired optimization).

        Uses bit-masking for fast k-mer encoding (2 bits per base).
        """
        self.kmer_hash = {}  # hash -> list of positions
        if self.n < k:
            return

        # Encode first k-mer
        mask = (1 << (2 * k)) - 1  # k bases × 2 bits
        w = 0
        valid_count = 0

        for i in range(min(k, self.n)):
            base = self.text[i].upper()
            if base in self.BASE_TO_BITS:
                w = ((w << 2) | self.BASE_TO_BITS[base]) & mask
                valid_count += 1

        if valid_count == k:
            if w not in self.kmer_hash:
                self.kmer_hash[w] = []
            self.kmer_hash[w].append(0)

        # Rolling window
        for i in range(k, self.n):
            base = self.text[i].upper()
            if base in self.BASE_TO_BITS:
                w = ((w << 2) | self.BASE_TO_BITS[base]) & mask

                if w not in self.kmer_hash:
                    self.kmer_hash[w] = []
                self.kmer_hash[w].append(i - k + 1)

    def get_kmer_positions(self, kmer: str) -> List[int]:
        """Get positions of k-mer using hash table (O(1) lookup).

        Args:
            kmer: k-mer sequence (must be valid DNA bases)

        Returns:
            List of positions where k-mer occurs
        """
        if len(kmer) > 8 or not self.kmer_hash:
            # Fall back to FM-index for longer k-mers
            return self.locate_positions(kmer)

        # Encode k-mer to hash
        w = 0
        for base in kmer.upper():
            if base not in self.BASE_TO_BITS:
                return []
            w = (w << 2) | self.BASE_TO_BITS[base]

        return self.kmer_hash.get(w, [])

    def clear(self):
        """Release heavy memory structures to let GC reclaim memory."""
        # Replace large attributes with minimal stubs
        self.text = ""
        self.text_arr = np.array([], dtype=np.uint8)
        self.bwt_arr = np.array([], dtype=np.uint8)
        self.suffix_array = np.array([], dtype=np.int32)
        self.sampled_sa = {}
        self.occ_checkpoints = {}
        self.char_counts = {}
        self.char_totals = {}
        self.alphabet = []
        self.char_to_code = {}
        self.code_to_char = {}
        self.char_counts_code = {}
        self.char_totals_code = {}
    
    def _build_suffix_array(self) -> np.ndarray:
        """Build suffix array, preferring pydivsufsort (C backend) with a NumPy fallback.

        Fallback uses prefix-doubling with NumPy lexsort (significantly faster than
        Python list.sort + lambdas). Complexity ~O(n log n) sorts.
        """
        # Prefer fast C implementation when available
        try:
            import pydivsufsort  # type: ignore
            sa_list = pydivsufsort.divsufsort(self.text_arr.tobytes())
            return np.array(sa_list, dtype=np.int32)
        except (ImportError, Exception):
            # Silently fall back to NumPy implementation
            pass

        n = self.n
        if n == 0:
            return np.array([], dtype=np.int32)

        # Initial rank from character codes
        codes = self.text_arr.astype(np.int32, copy=False)
        # Compress codes to 0..sigma-1 for stability
        uniq_codes, inv = np.unique(codes, return_inverse=True)
        rank = inv.astype(np.int32, copy=False)
        sa = np.arange(n, dtype=np.int32)

        k = 1
        tmp_rank = np.empty(n, dtype=np.int32)
        idx = np.arange(n, dtype=np.int32)
        while k < n:
            # secondary key is rank[i+k] else -1
            ipk = idx + k
            # Use safe indexing: clip ipk to valid range, then apply condition
            ipk_safe = np.clip(ipk, 0, n - 1)
            key2 = np.where(ipk < n, rank[ipk_safe], -1)
            # Sort by (rank[i], key2[i]) using lexsort with primary last
            sa = np.lexsort((key2, rank))
            # Compute new ranks
            r_sa = rank[sa]
            k2_sa = key2[sa]
            # mark changes
            change = np.empty(n, dtype=np.int32)
            change[0] = 0
            change[1:] = (r_sa[1:] != r_sa[:-1]) | (k2_sa[1:] != k2_sa[:-1])
            new_rank_ordered = np.cumsum(change, dtype=np.int32)
            # remap to original index order
            tmp_rank[sa] = new_rank_ordered
            rank, tmp_rank = tmp_rank, rank
            if rank[sa[-1]] == n - 1:
                break
            k <<= 1
        return sa.astype(np.int32, copy=False)
    
    def _build_bwt_array(self) -> np.ndarray:
        """Build BWT from suffix array as uint8 NumPy array (ASCII codes)."""
        if self.n == 0:
            return np.array([], dtype=np.uint8)
        sa = self.suffix_array.astype(np.int64, copy=False)
        # previous index (sa-1) % n
        prev_idx = (sa - 1) % self.n
        # Gather from numeric text array
        return self.text_arr[prev_idx]
    
    def _build_char_counts(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Count character frequencies and compute cumulative counts C[char]."""
        totals: Dict[str, int] = {c: 0 for c in self.alphabet}
        for ch in self.text:
            totals[ch] += 1
        counts: Dict[str, int] = {}
        cumulative = 0
        for char in self.alphabet:
            counts[char] = cumulative
            cumulative += totals[char]
        return counts, totals
    
    def _build_occurrence_checkpoints(self) -> Dict[int, np.ndarray]:
        """Build checkpointed occurrence counts for efficient rank queries with low memory.

        Returns a mapping from ASCII code -> np.ndarray of counts at positions m*k
        (prefix length), with cp[0] = 0. If the last block is partial, a final
        checkpoint with the total count at n is appended to mirror previous behavior.
        """
        bwt = self.bwt_arr
        n = bwt.size
        k = int(self.occ_sample_rate)
        if n == 0:
            return {}

        checkpoints: Dict[int, np.ndarray] = {}
        # Precompute full cumsum once per distinct code as we have small alphabets
        distinct_codes = np.unique(bwt)
        # indices where boundaries end (1-based length m*k corresponds to index m*k-1)
        block_ends = np.arange(k - 1, n, k, dtype=np.int64)

        for code in distinct_codes.tolist():
            mask = (bwt == code)
            csum = np.cumsum(mask, dtype=np.int32)
            # cp[0]=0, then take counts at each block end
            cp_list = [0]
            if block_ends.size:
                cp_list.extend(csum[block_ends].tolist())
            # Optionally append final count for partial block remainder
            if n % k != 0:
                cp_list.append(int(csum[-1]))
            checkpoints[int(code)] = np.asarray(cp_list, dtype=np.int32)
        # Ensure every alphabet character has a checkpoint array (even if absent)
        for c in self.alphabet:
            code = ord(c)
            if code not in checkpoints:
                # Build an all-zeros checkpoint array of same length as others
                # Determine representative length from any existing array
                any_cp = next(iter(checkpoints.values())) if checkpoints else np.array([0], dtype=np.int32)
                checkpoints[code] = np.zeros_like(any_cp)
        return checkpoints
    
    def _sample_suffix_array(self) -> Dict[int, int]:
        """Sample suffix array positions for space-efficient locating."""
        sampled = {}
        for i in range(0, self.n, self.sa_sample_rate):
            sampled[i] = self.suffix_array[i]
        return sampled
    
    def rank(self, char: Union[str, int], pos: int) -> int:
        """Count occurrences of `char` in bwt[0:pos]. Vectorized with checkpoints.

        Args:
            char: character (str) or ASCII code (int) to count
            pos: count occurrences in bwt[0:pos] (pos can be 0..n)
        """
        if pos <= 0:
            return 0
        if pos > self.n:
            pos = self.n
        code = ord(char) if isinstance(char, str) else int(char)
        cp = self.occ_checkpoints.get(code)
        if cp is None:
            return 0
        k = int(self.occ_sample_rate)
        cp_idx = pos // k
        cp_pos = cp_idx * k
        base = int(cp[cp_idx])
        # Fast remainder scan (Numba-accelerated if available)
        if pos > cp_pos:
            base += int(_count_equal_range(self.bwt_arr, cp_pos, pos, code))
        return base
    
    def backward_search(self, pattern: str) -> Tuple[int, int]:
        """
        Find suffix array interval for pattern using backward search.
        
        Returns:
            (start, end) interval in suffix array, or (-1, -1) if not found
        """
        if not pattern:
            return (0, self.n - 1)
        
        # Initialize with character range
        char = pattern[-1]
        if char not in self.char_counts:
            return (-1, -1)
        # sp inclusive, ep inclusive
        sp = self.char_counts[char]
        ep = sp + self.char_totals[char] - 1
        
        # Process pattern right to left
        for i in range(len(pattern) - 2, -1, -1):
            char = pattern[i]
            if char not in self.char_counts:
                return (-1, -1)

            sp = self.char_counts[char] + self.rank(char, sp)
            ep = self.char_counts[char] + self.rank(char, ep + 1) - 1

            if sp > ep:
                return (-1, -1)
        
        return (sp, ep)
    
    def count_occurrences(self, pattern: str) -> int:
        """Count pattern occurrences in text."""
        sp, ep = self.backward_search(pattern)
        if sp == -1:
            return 0
        return ep - sp + 1
    
    def locate_positions(self, pattern: str) -> List[int]:
        """
        Locate all positions of pattern in text.
        Uses sampled suffix array for efficiency.
        """
        sp, ep = self.backward_search(pattern)
        if sp == -1:
            return []

        # Directly read positions from the suffix array (much faster than LF walking)
        positions = self.suffix_array[sp:ep + 1].tolist()
        positions.sort()
        return positions
    
    def _get_suffix_position(self, sa_index: int) -> int:
        """Recover original text position from SA index using sampling."""
        if sa_index in self.sampled_sa:
            return self.sampled_sa[sa_index]
        
        # Walk using LF mapping until we hit a sampled position
        steps = 0
        current_idx = sa_index
        
        while current_idx not in self.sampled_sa:
            code = int(self.bwt_arr[current_idx])
            current_idx = self.char_counts_code[code] + self.rank(code, current_idx)
            steps += 1
        
        return (self.sampled_sa[current_idx] + steps) % self.n
