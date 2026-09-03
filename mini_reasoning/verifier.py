"""Verifier: extraction and equivalence grading, shared by all three roles.

Evaluation, reward and validation use the same verify_answer and the same
extraction machinery. They differ only in extraction strictness, which the
caller passes per call. The full record of what was decided and rejected is
in docs/design-notes.md.

- Boxed-only is the default, which is the strictness the reward role
  depends on: omitting the fallback argument fails toward strict rather
  than away from it.
- The lenient mode exists for comparability with officially graded numbers.
  The lab measured its absence at 8/350 samples lost, worth 4 accuracy
  points at best-of-N under official grading.
- Grading is sympy equivalence, proven equal to the official grader on
  every shared candidate in the lab cross-check. A None candidate is always
  False, and unparseable input grades False rather than raising, so a
  malformed sample cannot crash the reward path mid-training.
"""

import re
from tokenize import TokenError

from sympy import simplify
from sympy.core.sympify import SympifyError
from sympy.parsing import sympy_parser as spp
from sympy.polys.polyerrors import PolynomialError

_FALLBACK_MODES = ("number",)

_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_final_candidate(
    text: str,
    fallback: str | None = None,
) -> str | None:
    """Extracts the model's final answer from a sampled solution.

    The strict rule is the last complete ``\\boxed{...}``. Every way of
    failing it collapses to a single exit, so `fallback` alone decides
    whether a sample without a usable box has an answer at all.

    Args:
        text: The full sampled solution.
        fallback: Lenient mode to apply when no boxed answer is found.
            None, the default, is boxed-only, which is the strictness the
            reward role depends on.

    Returns:
        The extracted answer, or None when no rule matched.

    Raises:
        ValueError: If `fallback` names a mode that does not exist.
    """
    boxed = _extract_boxed(text)
    if boxed:
        return boxed
    return _rescue(text, fallback)


def _extract_boxed(text: str) -> str | None:
    """Finds the last complete ``\\boxed{...}`` in a sampled solution.

    Brace counting rather than a regular expression, because the contents
    nest. All three ways of failing return None: no ``\\boxed`` at all, no
    brace after it, and a brace the sample ends before closing.

    Args:
        text: The full sampled solution.

    Returns:
        The boxed contents stripped of surrounding whitespace, or None.
    """
    boxed_start_idx = text.rfind(r"\boxed")
    if boxed_start_idx == -1:
        return None

    current_idx = boxed_start_idx + len(r"\boxed")

    while current_idx < len(text) and text[current_idx].isspace():
        current_idx += 1

    if current_idx >= len(text) or text[current_idx] != "{":
        return None

    current_idx += 1
    brace_depth = 1
    content_start_idx = current_idx

    while current_idx < len(text) and brace_depth > 0:
        char = text[current_idx]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        current_idx += 1

    if brace_depth != 0:
        return None

    return text[content_start_idx:current_idx-1].strip()


def _rescue(text: str, fallback: str | None) -> str | None:
    """Applies the lenient rule where the strict one found nothing.

    Args:
        text: The full sampled solution.
        fallback: The mode to apply, or None for boxed-only strictness.

    Returns:
        The rescued answer, or None when the mode is None or nothing
        matched.

    Raises:
        ValueError: If `fallback` is neither None nor a known mode. This is
            checked before the text is searched, so a misspelled mode fails
            on every sample rather than only on those holding no number.
    """
    if fallback is None:
        return None

    if fallback not in _FALLBACK_MODES:
        raise ValueError(
            f"unknown fallback mode {fallback!r}: expected None (boxed-only) "
            f"or one of {', '.join(map(repr, _FALLBACK_MODES))}")

    numbers = _NUMBER_RE.findall(text)
    if not numbers:
        return None
    return numbers[-1].replace(",", "")


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
