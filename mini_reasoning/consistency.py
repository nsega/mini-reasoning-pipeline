"""Self-consistency voting over sampled solutions.

Ports the lab's module 02 with one policy change built in rather than
patched at the harness: None candidates are excluded before the vote.

- The lab measured the count-None design at up to 6/50 problems lost per
  n, because failed extractions pool on a single key while wrong answers
  scatter, and the None-counted curve went non-monotonic in n.
- The tiebreak is explicit and deterministic: first seen wins.
- Empty input, and input that is all None once filtered, raise rather than
  returning a sentinel. The lab chose loud errors over sentinel values.

Distinct from the vote-agreement stage in scorers.py, which returns a
per-candidate score under the scorer protocol rather than a winner. That
duplication is deliberate and recorded in docs/design-notes.md.
"""

import collections
from collections.abc import Sequence


def self_consistency_vote(candidates: Sequence[str | None]) -> str:
    """Returns the answer the most candidates agree on.

    Args:
        candidates: Extracted answers, with None wherever extraction found
            none.

    Returns:
        The most frequent answer among the non-None candidates. A tie goes
        to whichever of the tied answers appears first in `candidates`.

    Raises:
        ValueError: If `candidates` is empty, or holds nothing but None.
            Both are loud rather than sentinel-valued, because a caller
            that cannot tell a missing vote from a real one pools its
            failures on the sentinel.
    """
    if not candidates:
        raise ValueError("no candidates to vote on")

    present = [c for c in candidates if c is not None]
    if not present:
        raise ValueError("every candidate was None; nothing to vote on")

    counts = collections.Counter(present)
    top_count = max(counts.values())
    return next(c for c in present if counts[c] == top_count)
