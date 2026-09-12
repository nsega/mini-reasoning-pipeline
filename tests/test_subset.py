"""Contract tests for the committed evaluation subset.

The subset is generated once and committed, never regenerated
mid-experiment, so what these pin is the artifact itself rather than
the script that made it: that it is present, that it is the seed-42
draw whose skew the design notes quote, and that every problem carries
what evaluate requires of it.
"""
from pathlib import Path

import pytest

from mini_reasoning.evaluate import MATH500_LEVEL_COUNTS
from run_pipeline import build_parser, load_problems

SUBSET = Path(__file__).resolve().parents[1] / "data" / "math500_subset50.jsonl"


@pytest.fixture(scope="module")
def problems():
    return load_problems(SUBSET)


class TestPresence:
    """A fresh clone has to be able to run the default command."""

    def test_the_default_subset_is_the_committed_one(self):
        assert build_parser().parse_args([]).subset == str(
            SUBSET.relative_to(SUBSET.parents[1]))

    def test_it_holds_fifty_problems(self, problems):
        assert len(problems) == 50


class TestShape:
    """Every field evaluate reads, on every problem."""

    def test_each_problem_carries_what_evaluate_requires(self, problems):
        for problem in problems:
            assert {"unique_id", "problem", "answer", "level"} <= set(problem)

    def test_every_level_is_one_evaluate_can_weight(self, problems):
        levels = {p["level"] for p in problems}
        assert levels <= set(MATH500_LEVEL_COUNTS)

    def test_the_problems_are_distinct(self, problems):
        assert len({p["unique_id"] for p in problems}) == 50


class TestIdentity:
    """The draw itself, not just a well-formed file."""

    def test_the_source_indices_are_sorted(self, problems):
        indices = [p["source_index"] for p in problems]
        assert indices == sorted(indices)
        assert len(set(indices)) == 50

    def test_the_mean_level_is_the_documented_skew(self, problems):
        """3.88 against 3.44 for the full set is why the reweighted
        accuracy exists at all, and it identifies this exact draw."""
        mean = sum(p["level"] for p in problems) / len(problems)
        assert mean == pytest.approx(3.88, abs=0.005)
