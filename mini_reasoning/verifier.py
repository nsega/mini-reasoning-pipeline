"""Verifier: extraction + equivalence grading. Role 1 of 3 (evaluation).

To be self-written by Naoki (scaffold stub; contract below encoded in
tests/test_verifier.py). Lab lessons to carry in:

- extract_final_candidate(text, fallback=None) -> str | None
  Last \\boxed{...} via brace counting (regex cannot nest). NEW vs the
  lab version: an optional fallback mode, because the lab measured the
  no-fallback cost precisely (8/350 samples lost vs the official
  number_then_full fallback; at best-of-N selection time the missing
  fallback was worth 4 accuracy points under official grading). Naoki
  designs the fallback contract; boxed-only (fallback=None) must remain
  the default so the REWARD role stays strict.
- verify_answer(candidate, ground_truth) -> bool
  Sympy equivalence (simplify(gt - pred) == 0 shape); proven equivalent
  to the official grader on every shared candidate in the lab
  cross-check. None is always False. The lab's normalize_text was
  Claude-written and must be rewritten from scratch here.
"""


def extract_final_candidate(text, fallback=None):
    raise NotImplementedError("capstone self-written implementation (Naoki)")


def verify_answer(candidate, ground_truth):
    raise NotImplementedError("capstone self-written implementation (Naoki)")
