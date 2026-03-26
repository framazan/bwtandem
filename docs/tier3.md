# Tier 3: Long-Read & Structural Repeat Finder

## 1. Introduction: The Challenge of Macro-Satellites

In the landscape of genomic repetitive elements, "Tier 3" targets the giants: **macro-satellites** and **structural repeats**. While microsatellites (1-6bp units) are ubiquitous and relatively easy to find using sliding windows, macro-satellites present a unique set of computational challenges that render standard approaches ineffective.

### 1.1. The Biological Problem
Macro-satellites, such as centromeric alpha-satellites (171bp units in humans) or ribosomal RNA gene arrays (multi-kilobase units), span vast genomic regions. They are characterized by:
*   **Extreme Length**: Arrays can reach megabases in size.
*   **Complex Evolution**: Unlike microsatellites which evolve via polymerase slippage, macro-satellites evolve through unequal crossing over and gene conversion. This results in "higher-order repeats" (HORs) where the repeating unit itself is a mosaic of smaller subunits.
*   **Sequence Divergence**: Copies within an array can diverge significantly (20-30%), making exact string matching useless.

### 1.2. The Computational Bottleneck
A naive sliding window approach, effective for short repeats, fails here. To detect a 100bp repeat with a sliding window, one must compare a 100bp window against its neighbor at every position. As the window size grows, the computational cost explodes ($O(N \times W)$). Furthermore, structural variations (insertions of transposons within arrays) break the contiguity required by local alignment tools.

**Tier 3 solves this by inverting the problem.** Instead of asking "is this local window repetitive?", it asks "where else in the genome does this sequence occur?". This requires a global index.

---

## 2. System Architecture & Plumbing

The Tier 3 pipeline is built on a **Seed-and-Extend** architecture, powered by a **Burrows-Wheeler Transform (BWT)** index. The data flow is designed to minimize Python-level operations in favor of vectorized NumPy operations and compiled Cython kernels.

### 2.1. High-Level Data Flow

1.  **The Oracle (BWT)**: The entire genome is indexed into a BWT. This allows us to query the global frequency and location of any k-mer in $O(k)$ time.
2.  **Sparse Sampling**: We scan the genome at a coarse stride (e.g., every 100bp).
3.  **Global Lookup**: For each sampled k-mer, we query the BWT to find *all* its occurrences in the genome.
4.  **1D Clustering**: We analyze the sorted list of occurrence positions to find periodic patterns (arithmetic progressions).
5.  **Seed Extension**: Valid periodic patterns become "seeds". We expand these seeds into full arrays using a SIMD-accelerated extension algorithm.
6.  **Refinement**: Raw arrays are polished using dynamic programming to determine exact boundaries and consensus motifs.

---

## 3. The Global Oracle: BWT & FM-Index

The heart of the plumbing is the `BWTCore` class (`src/bwt_core.py`). It provides the "God's eye view" of the genome.

### 3.1. The Burrows-Wheeler Transform
The BWT permutes the genome $T$ into a string $L$ (the Last column of the sorted rotation matrix). This transformation clusters similar contexts together. Crucially, it allows for the **FM-index**, a compressed data structure that supports fast pattern matching.

In our implementation:
*   **Alphabet Reduction**: We map DNA to 2-bit integers (A=0, C=1, G=2, T=3).
*   **Suffix Array (SA)**: We store the SA to map BWT rows back to genomic coordinates. To save memory, we sample the SA (default rate 32).
*   **Checkpoints**: To accelerate rank queries (counting characters in a prefix of $L$), we store pre-calculated counts every 128 bases.

### 3.2. The `locate_positions` Operation
This is the critical primitive. Given a query pattern $P$ (a k-mer):
1.  **Backward Search**: We iteratively update a range $[sp, ep]$ in the BWT rows that share the suffix $P$. This takes $O(|P|)$ steps using the LF-mapping property: $LF(i, c) = C[c] + Rank(c, i)$.
2.  **SA Lookup**: Once we have the range $[sp, ep]$, the size of the range tells us the count. To get the positions, we look up `SA[sp]` through `SA[ep]`.
    *   *Optimization*: If `SA[i]` is not stored (due to sampling), we step forward in the text (using the LF mapping) until we hit a sampled position, then adjust the coordinate.

This mechanism allows Tier 3 to instantly retrieve the coordinates of every copy of a repeat unit, regardless of how far apart they are.

---

## 4. The Sampling Strategy: A Signal Processing Approach

Tier 3 relies on a principle analogous to the **Nyquist-Shannon sampling theorem**.

### 4.1. The Sampling Theorem for Repeats
To detect a tandem repeat of length $L$, we do not need to verify every base pair. We only need to sample the sequence at a frequency sufficient to capture the "beat" of the repeat.

If we sample the genome with a stride $S$, any repeat array with total length $L_{array} > S + k$ (where $k$ is the k-mer size) is guaranteed to be sampled at least once.
*   **Default Stride**: 100bp.
*   **Implication**: We reduce the search space by 100x compared to a base-by-base scan.

### 4.2. K-mer Selection
At each sampled position $i$, we extract a k-mer (default 20bp).
*   **Why 20bp?**
    *   **Uniqueness**: In a random genome, a 20-mer occurs once every $4^{20} \approx 10^{12}$ bases. A specific 20-mer is highly unlikely to appear multiple times by chance.
    *   **Conservation**: In a repeat with 10% divergence, the probability of a 20-mer remaining unmutated is $(1-0.1)^{20} \approx 12\%$. While low, a long array will contain many such conserved islands.

---

## 5. Periodicity Detection: 1D Clustering

Once `locate_positions` returns a list of genomic coordinates $P = [p_1, p_2, \dots, p_m]$, we must determine if they form a tandem repeat. This transforms the biological problem into a numerical one: **finding arithmetic progressions in a sorted list**.

### 5.1. The Algorithm (`find_periodic_runs`)
Implemented in Cython (`src/_accelerators.pyx`), this algorithm scans the sorted positions in $O(m)$ time.

1.  **Differential Analysis**: We compute adjacent differences $d_j = p_{j+1} - p_j$.
2.  **Run Detection**: We look for runs of consistent differences.
    *   A run is defined by a `current_period` (running average of differences).
    *   A new difference $d$ extends the run if $|d - \text{current\_period}| < \text{tolerance}$.
    *   Tolerance is typically 3% of the period, allowing for small indels between copies.
3.  **Gap Handling**: The current implementation focuses on *adjacent* consistency. If a mutation destroys a k-mer, the gap becomes $2 \times \text{period}$. Future improvements could use GCD analysis to bridge these gaps, but currently, we rely on the array being dense enough to have contiguous conserved segments.

This step filters out random matches (which have random distances) and isolates the structured periodicity of tandem repeats.

---

## 6. The Extension Engine: SIMD & SWAR

A "seed" from the clustering step gives us a start position and a period. However, the repeat likely extends into regions where the seed k-mer is mutated. To find the true boundaries, we switch to a robust extension method.

### 6.1. The "SWAR" Optimization
We use a technique called **SWAR (SIMD Within A Register)** to calculate Hamming distances incredibly fast. This is implemented in `src/_accelerators.pyx`.

*   **Data Packing**: We pack the DNA sequence into 64-bit integers. Each base takes 2 bits, so one 64-bit integer holds 32 bases.
*   **XOR Difference**: To compare two 32-base sequences $A$ and $B$, we compute $X = A \oplus B$.
    *   If bases match, the result is `00`.
    *   If they differ, the result is non-zero (`01`, `10`, or `11`).
*   **Parallel Counting**: We need to count the non-zero 2-bit slots.
    *   We mask the result to isolate differences.
    *   We use the `popcount` (population count) CPU instruction to sum the set bits.

This allows us to compare sequences 32 bases at a time, achieving a theoretical speedup of 32x over byte-by-byte comparison.

### 6.2. Bidirectional Extension Logic
1.  **Consensus Definition**: The sequence at the seed position is treated as the temporary consensus.
2.  **Right Extension**: We step forward by `period` bases. We compare the new window to the consensus using the SWAR Hamming distance.
    *   **Threshold**: We allow a mismatch rate (e.g., 20%). If `mismatches < period * 0.2`, we accept the copy and continue.
3.  **Left Extension**: We repeat the process backwards.

This "greedy" extension is extremely fast and robust to point mutations, though it stops at large insertions/deletions (indels).

---

## 7. Refinement & Consensus: The Final Polish

The extension phase gives us a rough region $[start, end]$ and a raw motif. The final stage (`src/motif_utils.py`) refines this into a high-quality annotation.

### 7.1. Boundary Trimming
The greedy extension might overshoot or undershoot slightly.
*   **Anchor-Based Scan**: For massive arrays, we re-verify the boundaries by scanning outwards from the seed using a stricter 75% identity check. This ensures the reported region is high-quality.

### 7.2. Consensus Building
We extract the individual units from the array and perform a **Multiple Sequence Alignment (MSA)** approximation.
1.  **Stacking**: We conceptually stack the units on top of each other.
2.  **Majority Vote**: At each column of the stack, we determine the most frequent base.
3.  **Result**: This yields the "consensus motif"—the idealized sequence of the repeat unit.

### 7.3. TRF Statistics
Finally, we calculate standard metrics to make the output compatible with Tandem Repeats Finder (TRF):
*   **Score**: A weighted sum of matches (+2) and mismatches (-7).
*   **Entropy**: Shannon entropy of the motif (measure of complexity).
*   **Composition**: %A, %C, %G, %T.

---

## 8. Performance & Complexity Analysis

### 8.1. Time Complexity
Let $N$ be the genome size, $S$ the sampling stride, and $M$ the average frequency of a k-mer.

1.  **Sampling Loop**: Runs $N/S$ times.
2.  **BWT Lookup**: $O(k)$ per sample.
3.  **Clustering**: $O(M \log M)$ to sort positions (done by BWT implicitly) + $O(M)$ to scan.
4.  **Extension**: $O(L)$ where $L$ is the length of the found repeat.

Total Complexity: $\approx O(\frac{N}{S} \cdot M + \sum L_{repeats})$

In practice, $M$ is small for unique regions and capped (e.g., at 500) for repetitive elements to prevent explosion. This makes the algorithm effectively **linear** with respect to genome size, $O(N)$.

### 8.2. Memory Complexity
The memory footprint is dominated by the BWT index.
*   **Text Array**: $N$ bytes (uint8).
*   **Suffix Array**: $4 \times (N / 32)$ bytes (int32, sampled).
*   **Checkpoints**: $4 \times (N / 128) \times 4$ bytes (4 characters).

For a 100MB chromosome, the index requires roughly 150-200MB of RAM, making it feasible to run on standard laptops.

---

## 9. Summary

Tier 3 represents a shift from **local alignment** to **global indexing**. By leveraging the BWT, it transforms the search for long, complex repeats into a series of efficient integer operations (clustering) and bitwise logic (SWAR extension). This "plumbing" allows it to detect megabase-scale structures that are invisible to standard window-based tools, providing a comprehensive view of the genome's architectural skeleton.
