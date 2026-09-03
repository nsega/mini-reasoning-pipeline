"""Contract tests for the verifier (scaffold; implementations are Naoki's).

Week 1 pass criteria carried forward, plus the fallback contract the
lab measured the need for.
"""
import pytest

from mini_reasoning.verifier import extract_final_candidate, verify_answer


class TestExtraction:
    def test_simple_boxed(self):
        assert extract_final_candidate(r"so \boxed{42}") == "42"

    def test_nested_braces(self):
        assert extract_final_candidate(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_last_box_wins(self):
        text = r"first \boxed{1} then corrected: \boxed{2}"
        assert extract_final_candidate(text) == "2"

    def test_negative_and_whitespace(self):
        assert extract_final_candidate(r"\boxed{ -3 }") == "-3"

    def test_no_box_default_is_none(self):
        # boxed-only strictness is the DEFAULT: the reward role depends on it.
        assert extract_final_candidate("the answer is 7") is None


class TestFallback:
    def test_fallback_rescues_bare_number(self):
        # The lab measured 8/350 samples lost to no-fallback; the capstone
        # extractor gains an opt-in fallback. Exact mode names are Naoki's
        # design; this test pins only that SOME non-None fallback exists.
        got = extract_final_candidate("the answer is 7", fallback="number")
        assert got is not None and "7" in got


class TestFallbackContract:
    """The fallback decisions TestFallback above does not pin.

    The committed test asserts only that SOME non-None mode rescues a
    bare number. These pin the contract designed around it: which
    failures the rescue applies to, which number it picks, and what a
    bad mode name does. Recorded in docs/design-notes.md under
    "The fallback contract".
    """

    def test_boxed_wins_over_fallback(self):
        assert extract_final_candidate(
            r"\boxed{1}, or maybe 9", fallback="number") == "1"

    def test_no_number_stays_none(self):
        assert extract_final_candidate(
            "no digits at all", fallback="number") is None

    def test_last_number_wins(self):
        assert extract_final_candidate(
            "first 3 then 7", fallback="number") == "7"

    def test_negative_decimal_survives(self):
        assert extract_final_candidate(
            "the answer is -3.5", fallback="number") == "-3.5"

    def test_truncated_box_is_rescued(self):
        assert extract_final_candidate(
            r"\boxed{1} then \boxed{2", fallback="number") == "2"

    def test_truncated_box_stays_none_when_strict(self):
        assert extract_final_candidate(r"so \boxed{42") is None

    def test_empty_box_is_not_an_answer(self):
        assert extract_final_candidate(r"\boxed{}") is None

    def test_unknown_mode_raises_regardless_of_input(self):
        for text in ("the answer is 7", "no digits at all"):
            with pytest.raises(ValueError):
                extract_final_candidate(text, fallback="numbers")


class TestVerification:
    def test_exact(self):
        assert verify_answer("42", "42")

    def test_equivalent_forms(self):
        assert verify_answer("0.5", r"\frac{1}{2}")

    def test_none_is_false(self):
        assert verify_answer(None, "42") is False

    def test_whitespace_tolerant(self):
        assert verify_answer("  42 ", "42")

    def test_wrong_is_false(self):
        assert verify_answer("41", "42") is False
