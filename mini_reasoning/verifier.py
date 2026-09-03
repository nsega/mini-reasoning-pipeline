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

import re
from tokenize import TokenError

from sympy import simplify
from sympy.core.sympify import SympifyError
from sympy.parsing import sympy_parser as spp
from sympy.polys.polyerrors import PolynomialError


def extract_final_candidate(text, fallback=None):
    raise NotImplementedError("capstone self-written implementation (Naoki)")


def verify_answer(candidate: str | None, ground_truth: str) -> bool:
    """Grades an extracted answer against the ground truth.

    Shared unchanged by all three roles. Evaluation, reward and validation
    differ only in the extraction strictness that produced `candidate`.

    Args:
        candidate: An extracted answer, or None when extraction found none.
        ground_truth: The reference answer.

    Returns:
        True when the two are equivalent under sympy simplification, False
        otherwise. A None candidate is always False.
    """
    # None must be rejected BEFORE normalization: _normalize_text(None) == ""
    # would otherwise let a missing candidate match an empty ground truth
    if candidate is None or ground_truth is None:
        return False

    candidate = _normalize_text(candidate)
    ground_truth = _normalize_text(ground_truth)

    if ground_truth == candidate:
        return True

    gtruth, pred = _parse_expr(ground_truth), _parse_expr(candidate)

    if gtruth is not None and pred is not None:
        try:
            return simplify(gtruth - pred) == 0
        except (SympifyError, TypeError):
            pass

    return False


def _normalize_text(text: str | None) -> str:
    """Rewrites LaTeX answer forms into something sympy can parse.

    Args:
        text: An extracted answer, or None.

    Returns:
        The normalized answer, or the empty string for empty input.
    """
    # Minimal subset of the official normalize_text (Claude-written during
    # pairing): only what the current tests need. Rewrite before migrating
    # to the capstone.
    if not text:
        return ""
    text = text.strip().strip("$ ")
    # \frac{a}{b} -> (a)/(b); parentheses keep operator precedence intact
    # for compound numerators/denominators like \frac{1}{x+1}
    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        lambda m: f"({m.group(1)})/({m.group(2)})",
        text,
    )
    return text.replace("{", "").replace("}", "").strip()


def _parse_expr(expr):
    """Parses a normalized answer, returning None if it will not parse.

    Args:
        expr: A normalized answer string, or None.

    Returns:
        The parsed sympy expression, or None when the input is missing,
        implausibly long, or unparseable. Returning None rather than
        raising is what keeps an unparseable sample from crashing the
        reward path mid-training.
    """
    if expr is None or len(expr) > 2000:
        return None

    try:
        return spp.parse_expr(
            expr,
            transformations=(
                *spp.standard_transformations,

                spp.implicit_multiplication_application,
            ),

            evaluate=True,
        )
    except (SympifyError, SyntaxError, TypeError, AttributeError,
            IndexError, TokenError, ValueError, PolynomialError):
        return None
