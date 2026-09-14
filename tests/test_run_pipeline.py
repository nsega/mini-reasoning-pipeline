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
import torch
from torch import nn

from mini_reasoning.scorers import (
    Candidate, Composite, LogprobRank, ParseabilityGate, VoteAgreement,
)
from mini_reasoning.trainer import Rollouts
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


class TinyPolicy(nn.Module):
    """A real, tiny causal LM, so the training stage trains something."""

    def __init__(self, vocab=32, dim=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, input_ids):
        return type("Out", (), {"logits": self.head(self.embed(input_ids))})


def loader(name):
    torch.manual_seed(0)
    return TinyPolicy(), object()


def canned_rollout(model, tokenizer, problem, n):
    torch.manual_seed(0)
    texts = [r"so \boxed{42}", r"so \boxed{41}", r"so \boxed{42}"][:n]
    return Rollouts(sequences=torch.randint(0, 32, (len(texts), 7)),
                    prompt_len=3, texts=texts)


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

    def test_a_missing_subset_names_the_command_that_rebuilds_it(
            self, tmp_path):
        """The file is absent from a fresh clone: the error has to say
        how to make it, or the one-command run dead-ends. Naming the
        script alone is not enough, which is why this matches the whole
        command: datasets sits in its own dependency group, so an
        invocation without --group data dead-ends just as surely, on
        ModuleNotFoundError instead."""
        with pytest.raises(
                FileNotFoundError,
                match=r"--group data python data/make_subset\.py"):
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

    def test_training_runs_and_is_followed_by_the_validation_eval(
            self, tmp_path):
        """The whole point of the stage: a second eval on the same
        subset with the same ruler, taken after the policy moved."""
        args = args_for(tmp_path, **{"--steps": 2})
        results = run(args, load_model=loader, sampler=canned,
                      rollout=canned_rollout)
        assert len(results["training"]["steps"]) == 2
        assert results["validation"]["accuracy"] == pytest.approx(0.5)

    def test_the_training_history_is_written_out(self, tmp_path):
        args = args_for(tmp_path, **{"--steps": 1})
        run(args, load_model=loader, sampler=canned, rollout=canned_rollout)
        out = tmp_path / "results" / "training.json"
        assert json.loads(out.read_text())["steps"][0]["unique_id"] == "p1"


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

    def test_the_learning_rate_is_settable(self):
        assert build_parser().parse_args(["--lr", "1e-5"]).lr == 1e-5
