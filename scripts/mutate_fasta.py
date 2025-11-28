#!/usr/bin/env python3
import sys
import random
import shutil
from pathlib import Path

def mutate_fasta(path, fraction=0.01, seed=None):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"File not found: {path}")
    if seed is not None:
        random.seed(seed)

    backup = p.with_suffix(p.suffix + '.bak')
    shutil.copy(p, backup)

    with p.open('r', encoding='utf-8') as fh:
        lines = fh.read().splitlines(keepends=True)

    # Map global sequence positions to (line_idx, char_idx)
    pos_map = []  # list of (line_idx, char_idx)
    for li, line in enumerate(lines):
        if line.startswith('>'):
            continue
        for ci, ch in enumerate(line):
            if ch in 'ACGTacgt':
                pos_map.append((li, ci))

    total = len(pos_map)
    if total == 0:
        print('No A/C/G/T bases found to mutate.')
        return 0

    num_mut = max(1, int(round(total * fraction)))
    picks = set(random.sample(range(total), num_mut))

    # Convert sequence lines to mutable lists
    line_lists = [list(l) for l in lines]

    bases = ['A','C','G','T']
    mutated = 0
    for idx in picks:
        li, ci = pos_map[idx]
        orig = line_lists[li][ci]
        orig_up = orig.upper()
        choices = [b for b in bases if b != orig_up]
        newb = random.choice(choices)
        # Preserve case of original
        if orig.islower():
            newb = newb.lower()
        line_lists[li][ci] = newb
        mutated += 1

    # Write back
    with p.open('w', encoding='utf-8') as fh:
        fh.write(''.join(''.join(ll) for ll in line_lists))

    print(f'Backup written to: {backup}')
    print(f'Total A/C/G/T bases: {total}')
    print(f'Mutated bases: {mutated} ({mutated/total*100:.4f}%)')
    return mutated

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: mutate_fasta.py <fasta-file> [fraction] [seed]')
        raise SystemExit(1)
    path = sys.argv[1]
    fraction = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.01
    seed = int(sys.argv[3]) if len(sys.argv) >= 4 else None
    mutate_fasta(path, fraction=fraction, seed=seed)
