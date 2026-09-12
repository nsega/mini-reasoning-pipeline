"""Contract tests for the entry point's wiring.

The stages are the modules' own, already pinned in their own files.
What is pinned here is the wiring: which subset is read and what a
missing one says, which scorer the pipeline composes, that the
self-consistency stage re-reads the stored records rather than
generating again, that training is gated, and what is written out.

Model loading is the one step no test can reach, so it is injected,
as sampling is in tests/test_evaluate.py.
"""
import json

import pytest

from mini_reasoning.scorers import (
    Candidate, Composite, LogprobRank, ParseabilityGate, VoteAgreement,
)
from run_pipeline import (
    build_parser, build_scorer, load_problems, run, vote_accuracy,
)

P1 = {"unique_id": "p1", "problem": "What is 6*7?", "answer": "42",
      "level": 1}
P2 = {"unique_id": "p2", "problem": "What is 2+2?", "answer": "4",
      "level": 5}

SAMPLES = {
    "What is 6*7?": [
        Candidate(text=r"so \boxed{42}", logprob_summary=-1.0),
        Candidate(text=r"also \boxed{42}", logprob_summary=-0.1),
        Candidate(text=r"but \boxed{41}", logprob_summary=-2.0),
    ],
    "What is 2+2?": [
        Candidate(text=r"\boxed{5}", logprob_summary=-1.0),
        Candidate(text=r"\boxed{5}", logprob_summary=-0.5),
        Candidate(text=r"\boxed{4}", logprob_summary=-0.1),
    ],
}


def canned(model, tokenizer, problem, n):
    return list(SAMPLES[problem][:n])


def loader(name):
    return object(), object()


def subset_file(tmp_path, problems=(P1, P2)):
    path = tmp_path / "subset.jsonl"
    path.write_text("".join(json.dumps(p) + "\n" for p in problems))
    return path


def args_for(tmp_path, **overrides):
    argv = ["--subset", str(subset_file(tmp_path)),
            "--out-dir", str(tmp_path / "results"), "--n-samples", "3"]
    for flag, value in overrides.items():
        argv += [flag] if value is True else [flag, str(value)]
    return build_parser().parse_args(argv)


class TestLoadProblems:
    """The subset is read as one problem per line."""

    def test_reads_every_line_in_order(self, tmp_path):
        got = load_problems(subset_file(tmp_path))
        assert [p["unique_id"] for p in got] == ["p1", "p2"]

    def test_a_missing_subset_names_the_generator(self, tmp_path):
        """The file is absent from a fresh clone: the error has to say
        how to make it, or the one-command run dead-ends."""
        with pytest.raises(FileNotFoundError, match="make_subset"):
            load_problems(tmp_path / "absent.jsonl")

    def test_an_empty_subset_raises(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        with pytest.raises(ValueError):
            load_problems(empty)


class TestScorer:
    """The lab's composition verdict, lenient at both extraction points."""

    def test_the_stages_are_the_gate_then_rank_then_agreement(self):
        scorer = build_scorer(fallback="number")
        assert isinstance(scorer, Composite)
        assert [type(s) for s in scorer.stages] == [
            ParseabilityGate, LogprobRank, VoteAgreement]

    def test_both_extracting_stages_carry_the_callers_fallback(self):
        gate, _, agreement = build_scorer(fallback="number").stages
        assert gate.fallback == "number"
        assert agreement.fallback == "number"


class TestSelfConsistency:
    """Voting re-reads the stored records instead of generating again."""

    def test_votes_over_the_stored_candidates(self):
        records = [{"candidates": ["42", "42", "41"], "answer": "42"},
                   {"candidates": ["5", "5", "4"], "answer": "4"}]
        assert vote_accuracy(records) == pytest.approx(0.5)

    def test_a_record_with_nothing_extractable_counts_wrong(self):
        """self_consistency_vote raises on all-None by module policy, and
        a problem nothing could be extracted from is simply not correct."""
        records = [{"candidates": [None, None], "answer": "42"}]
        assert vote_accuracy(records) == 0.0


class TestTrainingGate:
    """Training is the one stage with no module behind it yet."""

    def test_skip_training_runs_the_baseline_and_the_vote(self, tmp_path):
        results = run(args_for(tmp_path, **{"--skip-training": True}),
                      load_model=loader, sampler=canned)
        assert results["baseline"]["accuracy"] == pytest.approx(0.5)
        assert results["vote_accuracy"] == pytest.approx(0.5)
        assert "validation" not in results

    def test_training_raises_until_the_harness_lands(self, tmp_path):
        with pytest.raises(NotImplementedError, match="harness"):
            run(args_for(tmp_path), load_model=loader, sampler=canned)


class TestResults:
    """A run leaves the records behind, because re-analysis needs them."""

    def test_the_run_is_written_under_the_out_dir(self, tmp_path):
        args = args_for(tmp_path, **{"--skip-training": True})
        run(args, load_model=loader, sampler=canned)
        out = tmp_path / "results" / "baseline.json"
        written = json.loads(out.read_text())
        assert written["accuracy"] == pytest.approx(0.5)
        assert [r["unique_id"] for r in written["records"]] == ["p1", "p2"]


class TestCli:
    """Defaults the README and the design audit both name."""

    def test_defaults(self):
        args = build_parser().parse_args([])
        assert args.n_samples == 7
        assert args.steps == 40
        assert args.subset == "data/math500_subset50.jsonl"

    def test_the_seed_is_settable_and_defaults_to_the_frozen_one(self):
        assert build_parser().parse_args([]).seed == 42
