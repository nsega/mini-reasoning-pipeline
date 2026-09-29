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
from pathlib import Path

import pytest
import torch
from torch import nn

from mini_reasoning.scorers import (
    Candidate, Composite, LogprobRank, ParseabilityGate, VoteAgreement,
)
from mini_reasoning.trainer import Rollouts
from run_pipeline import (
    build_parser, build_scorer, load_problems, run, split_held_out,
    vote_accuracy,
)

ROOT = Path(__file__).resolve().parents[1]
COMMITTED_RUN = ROOT / "docs" / "runs" / "2026-09-21-float32"

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
    """Parsed args over the two-problem subset. --held-out is 0 because
    the default of ten would reserve both problems."""
    argv = ["--subset", str(subset_file(tmp_path)),
            "--out-dir", str(tmp_path / "results"), "--n-samples", "3",
            "--held-out", "0"]
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
        assert results["training"]["updates"] == 2
        assert results["validation"]["accuracy"] == pytest.approx(0.5)

    def test_the_training_history_is_written_out(self, tmp_path):
        args = args_for(tmp_path, **{"--steps": 1})
        run(args, load_model=loader, sampler=canned, rollout=canned_rollout)
        out = tmp_path / "results" / "training.json"
        assert json.loads(out.read_text())["steps"][0]["unique_id"] == "p1"

    def test_a_trained_run_reports_by_group(self, tmp_path):
        """Training on the eval subset makes one blended number mix what
        training touched with what it never saw, so the run splits it."""
        args = args_for(tmp_path, **{"--steps": 1})
        results = run(args, load_model=loader, sampler=canned,
                      rollout=canned_rollout)
        written = json.loads(
            (tmp_path / "results" / "report.json").read_text())
        assert written == results["report"]
        assert written["groups"]["untouched"] == ["p2"]

    def test_an_untrained_run_writes_no_report(self, tmp_path):
        run(args_for(tmp_path, **{"--skip-training": True}),
            load_model=loader, sampler=canned)
        assert not (tmp_path / "results" / "report.json").exists()


class TestResults:
    """A run leaves the records behind, because re-analysis needs them."""

    def test_the_run_is_written_under_the_out_dir(self, tmp_path):
        args = args_for(tmp_path, **{"--skip-training": True})
        run(args, load_model=loader, sampler=canned)
        out = tmp_path / "results" / "baseline.json"
        written = json.loads(out.read_text())
        assert written["accuracy"] == pytest.approx(0.5)
        assert [r["unique_id"] for r in written["records"]] == ["p1", "p2"]


class TestReuseBaseline:
    """A stored baseline can stand in for the eval that produced it.

    The eval is deterministic in its seed, so a run whose training
    stage died reproduces the same records at the same cost. What the
    file cannot prove is the model and dtype behind it, so the checks
    here pin what is checkable and the flag asserts the rest.
    """

    def test_the_stored_baseline_is_read_instead_of_sampled(self, tmp_path):
        run(args_for(tmp_path, **{"--skip-training": True}),
            load_model=loader, sampler=canned)

        def forbidden(*args, **kwargs):
            raise AssertionError("the baseline was sampled again")

        results = run(args_for(tmp_path, **{"--skip-training": True,
                                            "--reuse-baseline": True}),
                      load_model=loader, sampler=forbidden)
        assert results["baseline"]["accuracy"] == pytest.approx(0.5)
        assert results["vote_accuracy"] == pytest.approx(0.5)

    def test_a_missing_baseline_says_how_to_produce_one(self, tmp_path):
        args = args_for(tmp_path, **{"--skip-training": True,
                                     "--reuse-baseline": True})
        with pytest.raises(FileNotFoundError,
                           match="without --reuse-baseline"):
            run(args, load_model=loader, sampler=canned)

    def test_a_baseline_from_other_problems_is_refused(self, tmp_path):
        """Reusing across subsets would make the before and after halves
        of the comparison describe different problems."""
        run(args_for(tmp_path, **{"--skip-training": True}),
            load_model=loader, sampler=canned)
        one = tmp_path / "one.jsonl"
        one.write_text(json.dumps(P1) + "\n")
        args = build_parser().parse_args(
            ["--subset", str(one), "--out-dir", str(tmp_path / "results"),
             "--n-samples", "3", "--held-out", "0", "--reuse-baseline",
             "--skip-training"])
        with pytest.raises(ValueError, match="different problems"):
            run(args, load_model=loader, sampler=canned)

    def test_a_baseline_at_another_sample_count_is_refused(self, tmp_path):
        """--n-samples is the candidate set every stage sees, so a
        baseline drawn at another width is not this run's before."""
        run(args_for(tmp_path, **{"--skip-training": True}),
            load_model=loader, sampler=canned)
        args = build_parser().parse_args(
            ["--subset", str(subset_file(tmp_path)),
             "--out-dir", str(tmp_path / "results"),
             "--n-samples", "2", "--held-out", "0", "--reuse-baseline",
             "--skip-training"])
        with pytest.raises(ValueError, match="samples per problem"):
            run(args, load_model=loader, sampler=canned)


class TestReusedBaselineProvenance:
    """The stored stamp is checked, not taken on trust."""

    def test_a_baseline_from_another_dtype_is_refused(self, tmp_path):
        """The bug this whole guard exists for: transformers changed its
        default dtype under an unchanged pin, so the same command built
        a different model. A before half from that model is not this
        run's before half."""
        run(args_for(tmp_path, **{"--skip-training": True}),
            load_model=loader, sampler=canned)
        stored = tmp_path / "results" / "baseline.json"
        written = json.loads(stored.read_text())
        written["provenance"]["dtype"] = "torch.bfloat16"
        stored.write_text(json.dumps(written))
        args = args_for(tmp_path, **{"--skip-training": True,
                                     "--reuse-baseline": True})
        with pytest.raises(ValueError, match="dtype"):
            run(args, load_model=loader, sampler=canned)

    def test_a_baseline_without_a_stamp_is_refused(self, tmp_path):
        """Records written before stamping cannot be checked at all, and
        the repo's rule is that a missing argument fails toward
        strictness rather than away from it."""
        run(args_for(tmp_path, **{"--skip-training": True}),
            load_model=loader, sampler=canned)
        stored = tmp_path / "results" / "baseline.json"
        written = json.loads(stored.read_text())
        del written["provenance"]
        stored.write_text(json.dumps(written))
        args = args_for(tmp_path, **{"--skip-training": True,
                                     "--reuse-baseline": True})
        with pytest.raises(ValueError, match="no provenance"):
            run(args, load_model=loader, sampler=canned)


class TestCli:
    """Defaults the README and the design audit both name."""

    def test_reuse_baseline_defaults_off(self):
        assert build_parser().parse_args([]).reuse_baseline is False

    def test_defaults(self):
        args = build_parser().parse_args([])
        assert args.n_samples == 7
        assert args.steps == 40
        assert args.subset == "data/math500_subset50.jsonl"

    def test_the_seed_is_settable_and_defaults_to_the_frozen_one(self):
        assert build_parser().parse_args([]).seed == 42

    def test_the_learning_rate_is_settable(self):
        assert build_parser().parse_args(["--lr", "1e-5"]).lr == 1e-5

    def test_the_split_defaults_to_ten_and_the_draw_cap_to_unset(self):
        args = build_parser().parse_args([])
        assert args.held_out == 10 and args.max_draws is None

    def test_a_draw_cap_below_one_is_refused_at_parse_time(self):
        """Found in review: train() refused it, but only after the model
        load and the hour-long baseline eval. 0 is a plausible guess at
        "no cap", so it has to fail before anything runs."""
        for bad in ("0", "-5"):
            with pytest.raises(SystemExit):
                build_parser().parse_args(["--max-draws", bad])

    def test_steps_below_one_are_refused_at_parse_time(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--steps", "0"])


class TestHeldOutSplit:
    """The last N problems are reserved; only the rest are trained on."""

    def test_it_reserves_the_last_n(self):
        assert split_held_out([P1, P2], 1) == ([P1], [P2])

    def test_zero_reserves_nothing(self):
        """problems[-0:] is the whole list in Python, so the obvious slice
        holds out everything when asked to hold out nothing."""
        assert split_held_out([P1, P2], 0) == ([P1, P2], [])

    def test_a_negative_count_raises(self):
        with pytest.raises(ValueError, match="held-out"):
            split_held_out([P1, P2], -1)

    def test_a_count_that_leaves_nothing_to_train_on_raises(self):
        with pytest.raises(ValueError, match="nothing to train"):
            split_held_out([P1, P2], 2)

    def test_the_default_holds_out_what_the_committed_run_never_drew(self):
        """The committed run is this one's control only while the default
        reserves exactly the problems it left untouched. If the subset's
        order or the default moves, this is where it shows."""
        defaults = build_parser().parse_args([])
        problems = load_problems(ROOT / defaults.subset)
        _, held_out = split_held_out(problems, defaults.held_out)
        steps = json.loads((COMMITTED_RUN / "training.json")
                           .read_text())["steps"]
        drawn = {s["unique_id"] for s in steps}
        assert [p["unique_id"] for p in held_out] == [
            p["unique_id"] for p in problems if p["unique_id"] not in drawn]


def recording_rollout(seen):
    """canned_rollout, noting which problem statements training reached."""
    def rollout(model, tokenizer, problem, n):
        seen.append(problem)
        return canned_rollout(model, tokenizer, problem, n)
    return rollout


def all_wrong_rollout(model, tokenizer, problem, n):
    torch.manual_seed(0)
    return Rollouts(sequences=torch.randint(0, 32, (n, 7)), prompt_len=3,
                    texts=[r"\boxed{0}"] * n)


class TestHeldOutWiring:
    """The split reaches training and nothing else."""

    def test_training_never_sees_a_held_out_problem(self, tmp_path):
        seen = []
        run(args_for(tmp_path, **{"--steps": 3, "--held-out": 1}),
            load_model=loader, sampler=canned,
            rollout=recording_rollout(seen))
        assert set(seen) == {P1["problem"]}

    def test_both_evals_still_cover_every_problem(self, tmp_path):
        results = run(args_for(tmp_path, **{"--steps": 1, "--held-out": 1}),
                      load_model=loader, sampler=canned,
                      rollout=canned_rollout)
        for stage in ("baseline", "validation"):
            assert [r["unique_id"] for r in results[stage]["records"]] == [
                "p1", "p2"]

    def test_the_reserved_ids_are_written_with_the_history(self, tmp_path):
        run(args_for(tmp_path, **{"--steps": 1, "--held-out": 1}),
            load_model=loader, sampler=canned, rollout=canned_rollout)
        written = json.loads(
            (tmp_path / "results" / "training.json").read_text())
        assert written["held_out"] == ["p2"]
        assert {"updates", "draws", "stopped"} <= set(written)

    def test_the_report_carries_the_held_out_group(self, tmp_path):
        results = run(args_for(tmp_path, **{"--steps": 1, "--held-out": 1}),
                      load_model=loader, sampler=canned,
                      rollout=canned_rollout)
        assert results["report"]["groups"]["held_out"] == ["p2"]

    def test_a_bad_split_fails_before_the_model_loads(self, tmp_path):
        """An hour of baseline eval is the price of finding out late."""
        def never(name):
            raise AssertionError("the model loaded before the split was "
                                 "checked")
        with pytest.raises(ValueError, match="nothing to train"):
            run(args_for(tmp_path, **{"--held-out": 2}), load_model=never,
                sampler=canned)

    def test_a_pool_that_all_goes_flat_still_validates_and_reports(
            self, tmp_path):
        """Zero updates is a result, not a crash: the run still ends in a
        validation eval and a report saying nothing stepped."""
        results = run(args_for(tmp_path, **{"--steps": 2}),
                      load_model=loader, sampler=canned,
                      rollout=all_wrong_rollout)
        assert results["training"]["stopped"] == "pool_exhausted"
        assert results["report"]["groups"]["stepped"] == []
        assert "validation" in results

    def test_reuse_baseline_accepts_a_baseline_over_every_problem(
            self, tmp_path):
        """The split never touches the evals, so a baseline stored before
        it is still this run's before half."""
        run(args_for(tmp_path, **{"--skip-training": True}),
            load_model=loader, sampler=canned)
        results = run(args_for(tmp_path, **{"--reuse-baseline": True,
                                            "--held-out": 1,
                                            "--steps": 1}),
                      load_model=loader, sampler=canned,
                      rollout=canned_rollout)
        assert results["report"]["groups"]["held_out"] == ["p2"]


class TestSeedStreams:
    """Training draws from its own random stream, never the eval's."""

    def test_training_never_reuses_an_eval_seed(self, tmp_path):
        """Found in review. The evals seed problem i with seed + i, and
        training seeded draw d with seed + d through the same sampling
        call, so the first pass replayed the baseline's own samples and
        first-pass retirement became the pre-filter the design rejects."""
        eval_seeds, train_seeds = [], []

        def sampler(model, tokenizer, problem, n):
            eval_seeds.append(torch.initial_seed())
            return canned(model, tokenizer, problem, n)

        def rollout(model, tokenizer, problem, n):
            train_seeds.append(torch.initial_seed())
            return canned_rollout(model, tokenizer, problem, n)

        run(args_for(tmp_path, **{"--steps": 3}), load_model=loader,
            sampler=sampler, rollout=rollout)
        assert train_seeds and eval_seeds
        assert not set(eval_seeds) & set(train_seeds)


class TestSkipTrainingIgnoresTheSplit:
    """Deferred from #16's review: the split only concerns training."""

    def test_an_eval_only_run_accepts_a_subset_the_split_would_empty(
            self, tmp_path):
        """--held-out's default of ten would reserve all of a two-problem
        subset, which refused an eval-only smoke run although nothing
        trains."""
        results = run(args_for(tmp_path, **{"--skip-training": True,
                                            "--held-out": 10}),
                      load_model=loader, sampler=canned)
        assert results["baseline"]["accuracy"] == pytest.approx(0.5)

