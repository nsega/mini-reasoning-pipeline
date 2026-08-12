"""Self-consistency voting over sampled solutions.

To be self-written by Naoki (port of the lab's module 02, which is
self-written and migration-eligible). One policy change is now
evidence-backed and should be built in rather than harness-patched:

- None candidates are EXCLUDED before voting. The lab measured the
  count-None design at up to 6/50 problems lost per n (failed
  extractions pool on one Counter key while wrong answers scatter),
  and the None-counted curve went non-monotonic in n.
- Tiebreak stays explicit and deterministic (first-seen wins).
- Empty input (or all-None after filtering): explicit, documented
  behavior; the lab chose loud errors over sentinel values.
"""


def self_consistency_vote(candidates):
    raise NotImplementedError("capstone self-written implementation (Naoki)")
