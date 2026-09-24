"""Contract tests for the before/after report, split by training's reach.

The headline blended three different populations into one number: the
problems training stepped on, the ones it drew and skipped as flat, and
the ones it never drew. It also counted one selected answer per problem
where seven samples were stored. What is pinned here is the split, the
two per-sample graders and where they disagree, and that the report
can be rebuilt offline from a stored run.
"""
import json

import pytest

from mini_reasoning.report import (
    GROUPS, format_table, main, report, sample_scores, training_groups,
)


def record(uid, answer, texts, candidates):
    """A stored eval record, holding only the fields the report reads."""
    return {"unique_id": uid, "answer": answer,
            "samples": [{"text": t} for t in texts],
            "candidates": candidates}


def evaluation(*records):
    return {"records": list(records)}


def step(uid, stepped):
    return {"unique_id": uid, "stepped": stepped, "mean_reward": 0.0,
            "grad_norm": 0.0}


BOXED_42 = r"so \boxed{42}"
BOXED_41 = r"so \boxed{41}"


def scored(uid, right, of=4):
    """A problem with `right` of `of` samples boxed and correct."""
    texts = [BOXED_42] * right + [BOXED_41] * (of - right)
    return record(uid, "42", texts, ["42"] * right + ["41"] * (of - right))


class TestGroups:
    """Membership comes from the training history alone."""

    def test_stepped_flat_and_untouched_come_from_the_history(self):
        groups = training_groups(
            ["p1", "p2", "p3", "p4"],
            [step("p1", True), step("p2", False), step("p3", True)])
        assert groups == {"stepped": ["p1", "p3"], "flat": ["p2"],
                          "untouched": ["p4"]}

    def test_a_problem_stepped_on_any_draw_counts_as_stepped(self):
        """Steps cycle the subset, so past its length a problem is drawn
        twice. The weights moved on it once, which is what the group
        asks about."""
        groups = training_groups(["p1"], [step("p1", False),
                                          step("p1", True)])
        assert groups["stepped"] == ["p1"] and groups["flat"] == []

    def test_untouched_is_empty_when_every_problem_was_drawn(self):
        """Nothing is reserved: untouched is whatever training happened
        not to reach, and at enough steps that is nothing."""
        groups = training_groups(["p1", "p2"], [step("p1", True),
                                                step("p2", False)])
        assert groups["untouched"] == []

    def test_every_group_keeps_the_subset_order(self):
        groups = training_groups(["p3", "p1", "p2"],
                                 [step("p2", True), step("p3", True)])
        assert groups["stepped"] == ["p3", "p2"]


class TestSampleScores:
    """Two graders, one per role, over every stored sample."""

    def test_the_graders_disagree_on_an_unboxed_correct_sample(self):
        """The reason to report both: the eval's grader rescues a bare
        number the reward would have scored zero."""
        got = sample_scores([record("p1", "42", ["the answer is 42"],
                                    ["42"])])
        assert got["p1"] == {"eval": 1.0, "reward": 0.0}

    def test_a_boxed_correct_sample_satisfies_both(self):
        got = sample_scores([record("p1", "42", [BOXED_42], ["42"])])
        assert got["p1"] == {"eval": 1.0, "reward": 1.0}

    def test_scores_are_shares_of_all_samples_not_of_the_selected_one(
            self):
        assert sample_scores([scored("p1", 1)])["p1"]["eval"] == 0.25

    def test_an_unextractable_sample_scores_zero(self):
        got = sample_scores([record("p1", "42", ["no answer"], [None])])
        assert got["p1"] == {"eval": 0.0, "reward": 0.0}


class TestReport:
    """Before against after, per group and over the whole subset."""

    def test_a_group_moves_by_its_mean_per_sample_change(self):
        got = report(evaluation(scored("p1", 1)),
                     evaluation(scored("p1", 3)),
                     {"steps": [step("p1", True)]})
        stepped = got["comparisons"]["stepped"]["reward"]
        assert stepped["before"] == 0.25 and stepped["after"] == 0.75
        assert stepped["delta"] == pytest.approx(0.5)

    def test_an_all_zero_flat_group_reports_no_change(self):
        got = report(evaluation(scored("p1", 0)),
                     evaluation(scored("p1", 0)),
                     {"steps": [step("p1", False)]})
        flat = got["comparisons"]["flat"]["reward"]
        assert flat["delta"] == 0.0 and flat["p"] == 1.0

    def test_an_empty_group_reports_no_comparison_rather_than_zero(self):
        """A zero would read as a measured null; there is nothing here
        to have measured."""
        got = report(evaluation(scored("p1", 1)),
                     evaluation(scored("p1", 2)),
                     {"steps": [step("p1", True)]})
        untouched = got["comparisons"]["untouched"]
        assert untouched["n"] == 0 and untouched["reward"] is None

    def test_the_whole_subset_is_reported_beside_the_groups(self):
        got = report(evaluation(scored("p1", 1), scored("p2", 1)),
                     evaluation(scored("p1", 3), scored("p2", 1)),
                     {"steps": [step("p1", True)]})
        assert got["comparisons"]["all"]["n"] == 2
        assert got["comparisons"]["all"]["reward"]["delta"] == (
            pytest.approx(0.25))
        assert set(got["comparisons"]) == {"all", *GROUPS}

    def test_the_p_value_is_the_same_on_every_call(self):
        args = (evaluation(scored("p1", 1), scored("p2", 2)),
                evaluation(scored("p1", 3), scored("p2", 1)),
                {"steps": [step("p1", True), step("p2", True)]})
        first = report(*args)["comparisons"]["stepped"]["reward"]["p"]
        assert report(*args)["comparisons"]["stepped"]["reward"]["p"] == first

    def test_before_and_after_must_grade_the_same_problems(self):
        with pytest.raises(ValueError, match="same problems"):
            report(evaluation(scored("p1", 1)),
                   evaluation(scored("p2", 1)),
                   {"steps": []})


class TestTable:
    """The printed form a run ends with."""

    def test_every_group_is_named(self):
        got = format_table(report(evaluation(scored("p1", 1)),
                                  evaluation(scored("p1", 2)),
                                  {"steps": [step("p1", True)]}))
        for name in ("all", *GROUPS):
            assert name in got

    def test_an_empty_group_says_so(self):
        got = format_table(report(evaluation(scored("p1", 1)),
                                  evaluation(scored("p1", 2)),
                                  {"steps": [step("p1", True)]}))
        assert "none" in got.split("untouched", 1)[1].splitlines()[0]

    def test_each_empty_group_explains_itself_and_no_other(self):
        """Found by a real run: one message for every empty group told a
        run whose drawn problems all stepped that nothing was held out,
        while its untouched problem sat on the next line."""
        got = format_table(report(
            evaluation(scored("p1", 1), scored("p2", 1)),
            evaluation(scored("p1", 2), scored("p2", 1)),
            {"steps": [step("p1", True)]}))
        flat_line = next(line for line in got.splitlines()
                         if line.startswith("flat"))
        assert "held out" not in flat_line
        assert "every drawn problem stepped" in flat_line


class TestOffline:
    """A stored run is reported without re-running anything."""

    def test_the_report_is_written_beside_the_run(self, tmp_path):
        base = evaluation(scored("p1", 1))
        after = evaluation(scored("p1", 3))
        history = {"steps": [step("p1", True)]}
        for name, body in (("baseline", base), ("validation", after),
                           ("training", history)):
            (tmp_path / f"{name}.json").write_text(json.dumps(body))
        main([str(tmp_path)])
        written = json.loads((tmp_path / "report.json").read_text())
        assert written == report(base, after, history)

    def test_a_run_without_training_says_what_is_missing(self, tmp_path):
        (tmp_path / "baseline.json").write_text(
            json.dumps(evaluation(scored("p1", 1))))
        with pytest.raises(FileNotFoundError, match="validation.json"):
            main([str(tmp_path)])
