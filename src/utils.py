import re
from typing import List, Tuple

def natural_sort_key(value: str):
    """Return a tuple usable for natural sorting (e.g., chr2 before chr10)."""
    if value is None:
        return ()

    parts = re.split(r'(\d+)', str(value))
    key_parts: List[Tuple[int, object]] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key_parts.append((0, int(part)))
        else:
            key_parts.append((1, part.lower()))
    return tuple(key_parts)
