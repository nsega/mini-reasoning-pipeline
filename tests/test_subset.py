"""Contract tests for the committed evaluation subset.

The subset is generated once and committed, never regenerated
mid-experiment, so what these pin is the artifact itself rather than
the script that made it: that it is present, that it is the seed-42
draw whose skew the design notes quote, and that every problem carries
what evaluate requires of it.
"""
import hashlib
from pathlib import Path

import pytest

from mini_reasoning.evaluate import MATH500_LEVEL_COUNTS
from run_pipeline import build_parser, load_problems

SUBSET = Path(__file__).resolve().parents[1] / "data" / "math500_subset50.jsonl"

SEED42_INDICES = (
    3, 12, 13, 15, 16, 44, 47, 49, 52, 57, 71, 79, 81, 101, 110, 111,
    112, 114, 119, 125, 140, 142, 172, 174, 183, 194, 214, 216, 229,
    258, 279, 287, 301, 302, 308, 327, 332, 346, 357, 359, 366, 377,
    379, 388, 390, 412, 414, 445, 456, 490,
)
"""``sorted(random.Random(42).sample(range(500), 50))``.

The seed determines these fifty numbers completely, so they can be
written down here rather than re-derived: recomputing the draw in the
test would only prove ``random`` is deterministic, while the committed
list is what a regenerated file has to still agree with.
"""

SUBSET_SHA256 = (
    "a8502eb815df79b972326c727b39eafce300c7046c6a918c2ba4ff4023595e17")


@pytest.fixture(scope="module")
def problems():
    return load_problems(SUBSET)


class TestPresence:
    """A fresh clone has to be able to run the default command."""

    def test_the_default_subset_is_the_committed_one(self):
        assert build_parser().parse_args([]).subset == SUBSET.relative_to(
            SUBSET.parents[1]).as_posix()

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
    """The draw itself, not just a well-formed file.

    Freezing is the whole point of committing the subset, so these fail
    on any change to it — which is the intended behaviour, not a
    brittleness to work around. A deliberate re-draw updates the
    constants above in the same commit that updates the file, and the
    design notes' numbers with them.
    """

    def test_the_indices_are_the_seed_42_draw(self, problems):
        """What distinguishes this subset from any other well-formed
        fifty. Sortedness and distinctness follow from the constant, so
        they are not asserted separately: a problem swapped for another
        at the same level keeps both, and the skew below, and is caught
        only here."""
        assert tuple(p["source_index"] for p in problems) == SEED42_INDICES

    def test_the_file_is_byte_for_byte_the_frozen_artifact(self):
        """The indices pin which problems were drawn, not what they
        say. Regenerating against a later revision of the upstream
        dataset would draw the same fifty indices and could still hand
        back edited problems or answers, which every other test here
        would accept."""
        digest = hashlib.sha256(SUBSET.read_bytes()).hexdigest()
        assert digest == SUBSET_SHA256

    def test_the_mean_level_is_the_documented_skew(self, problems):
        """3.88 against 3.44 for the full set is why the reweighted
        accuracy exists at all."""
        mean = sum(p["level"] for p in problems) / len(problems)
        assert mean == pytest.approx(3.88, abs=0.005)
