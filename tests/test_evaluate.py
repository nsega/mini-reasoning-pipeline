"""Contract tests for evaluate (role 3, and role 1 before training).

Written before the implementation, from the stub docstring and the
obligations docs/design-notes.md places on evaluate: build the material
bundle, store the per-problem record, select by argmax over the scorer,
check the all-rejected case first, grade with the caller's fallback, and
report raw and level-reweighted accuracy. Sampling is the one model-bound
step, so it is injected; scorers and the verifier are real.
"""
from types import SimpleNamespace

import pytest
import torch

from mini_reasoning.evaluate import (
    _eos_ids, evaluate, sequence_logprob_summary,
    token_logprobs_from_logits,
)
from mini_reasoning.scorers import (
    REJECTED, Candidate, Composite, ParseabilityGate, VoteAgreement,
)

P1 = {"unique_id": "p1", "problem": "What is 6*7?", "answer": "42",
      "level": 1}
P2 = {"unique_id": "p2", "problem": "What is 2+2?", "answer": "4",
      "level": 5}

SAMPLES = {
    "What is 6*7?": [
        Candidate(text=r"so \boxed{42}", logprob_summary=-1.0),
        Candidate(text="the answer is 7", logprob_summary=-0.1),
        Candidate(text=r"therefore \boxed{42}", logprob_summary=-2.0),
    ],
    "What is 2+2?": [
        Candidate(text=r"\boxed{5}", logprob_summary=-1.0),
        Candidate(text=r"\boxed{5}", logprob_summary=-1.0),
        Candidate(text=r"\boxed{4}", logprob_summary=-0.5),
    ],
}

MODEL, TOKENIZER = object(), object()


def canned(samples=SAMPLES):
    """A sampler that answers from a table and records how it was called."""
    calls = []

    def sampler(model, tokenizer, problem, n):
        calls.append((model, tokenizer, problem, n))
        return list(samples[problem][:n])

    sampler.calls = calls
    return sampler


class RejectAll:
    """A gate that rejects every candidate."""

    def score(self, candidates):
        return [REJECTED] * len(candidates)


class Prefer:
    """A scorer that puts one index first and ties the rest."""

    def __init__(self, index):
        self.index = index

    def score(self, candidates):
        return [1.0 if i == self.index else 0.0
                for i in range(len(candidates))]


def lenient():
    return Composite((ParseabilityGate(fallback="number"),
                      VoteAgreement(fallback="number")))


class TestSampling:
    """The sampler is the seam: it gets the model and n, nothing else."""

    def test_sampler_receives_model_tokenizer_problem_and_n(self):
        sampler = canned()
        evaluate(MODEL, TOKENIZER, [P1], lenient(), n_samples=2,
                 sampler=sampler)
        assert sampler.calls == [(MODEL, TOKENIZER, "What is 6*7?", 2)]

    def test_zero_samples_raises(self):
        """0 == 0 satisfies the count check and all([]) is True, so an
        empty candidate set would report 0% rather than a bad flag."""
        with pytest.raises(ValueError):
            evaluate(MODEL, TOKENIZER, [P1], lenient(), n_samples=0,
                     sampler=canned())

    def test_a_sampler_returning_the_wrong_count_raises(self):
        """The candidate set is what n_samples names, so a sampler that
        returns a different count is a bug, not a quietly weaker eval."""
        def short(model, tokenizer, problem, n):
            return SAMPLES[problem][:1]

        with pytest.raises(ValueError):
            evaluate(MODEL, TOKENIZER, [P1], lenient(), n_samples=3,
                     sampler=short)


class TestSelection:
    """Selection is argmax over the scorer, checked for all-rejected."""

    def test_selects_the_argmax_of_the_scorer(self):
        results = evaluate(MODEL, TOKENIZER, [P2], Prefer(2), n_samples=3,
                           fallback="number", sampler=canned())
        record = results["records"][0]
        assert record["selected"] == 2
        assert record["correct"] is True

    def test_first_of_tied_maxima_wins(self):
        results = evaluate(MODEL, TOKENIZER, [P2], lenient(), n_samples=3,
                           fallback="number", sampler=canned())
        record = results["records"][0]
        assert record["scores"] == [1.0, 1.0, 0.0]
        assert record["selected"] == 0
        assert record["correct"] is False

    def test_all_rejected_selects_nothing_and_counts_wrong(self):
        """The composite returns all -inf and says nothing more; the
        caller must notice before argmax, or it silently picks index 0."""
        results = evaluate(MODEL, TOKENIZER, [P1], RejectAll(), n_samples=3,
                           fallback="number", sampler=canned())
        record = results["records"][0]
        assert record["selected"] is None
        assert record["correct"] is False
        assert results["accuracy"] == 0.0


class TestGrading:
    """Grading uses the caller's fallback, defaulting toward strict."""

    def test_fallback_rescues_an_unboxed_selection(self):
        results = evaluate(MODEL, TOKENIZER, [P1], Prefer(1), n_samples=3,
                           fallback="number", sampler=canned())
        record = results["records"][0]
        assert record["candidates"][1] == "7"
        assert record["correct"] is False

    def test_omitted_fallback_grades_boxed_only(self):
        results = evaluate(MODEL, TOKENIZER, [P1], Prefer(1), n_samples=3,
                           sampler=canned())
        record = results["records"][0]
        assert record["candidates"][1] is None
        assert record["correct"] is False


class TestRecord:
    """The record holds the bundle, so a composite can be re-run offline."""

    def test_record_stores_samples_candidates_scores_and_selection(self):
        results = evaluate(MODEL, TOKENIZER, [P1], lenient(), n_samples=3,
                           fallback="number", sampler=canned())
        record = results["records"][0]
        assert record["unique_id"] == "p1"
        assert record["level"] == 1
        assert record["answer"] == "42"
        assert record["samples"] == [
            {"text": r"so \boxed{42}", "logprob_summary": -1.0,
             "finished": None},
            {"text": "the answer is 7", "logprob_summary": -0.1,
             "finished": None},
            {"text": r"therefore \boxed{42}", "logprob_summary": -2.0,
             "finished": None},
        ]
        assert record["candidates"] == ["42", "7", "42"]
        assert record["scores"] == [1.0, 0.0, 1.0]
        assert record["selected"] == 0
        assert record["correct"] is True

    def test_stored_bundle_rescoring_reproduces_the_stored_scores(self):
        """The forensics decision rests on this: purity plus a stored
        bundle means the scorer can be re-run over the record."""
        scorer = lenient()
        results = evaluate(MODEL, TOKENIZER, [P1, P2], scorer, n_samples=3,
                           fallback="number", sampler=canned())
        for record in results["records"]:
            rebuilt = [Candidate(**sample) for sample in record["samples"]]
            assert scorer.score(rebuilt) == record["scores"]

    def test_a_sample_that_hit_the_cap_is_marked_unfinished(self):
        """Decoding strips the stop token, so without this flag a sample
        that ran out of tokens is indistinguishable offline from one
        that finished and simply got the answer wrong."""
        def capped(model, tokenizer, problem, n):
            return [
                Candidate(text=r"\boxed{42}", logprob_summary=-1.0,
                          finished=True),
                Candidate(text="ran out before the box",
                          logprob_summary=-2.0, finished=False),
            ]

        results = evaluate(MODEL, TOKENIZER, [P1], lenient(), n_samples=2,
                           fallback="number", sampler=capped)
        samples = results["records"][0]["samples"]
        assert [s["finished"] for s in samples] == [True, False]


class TestIncrementalRecords:
    """A caller can persist each record as it is finished."""

    def test_on_record_is_called_as_each_problem_finishes(self):
        seen = []
        evaluate(MODEL, TOKENIZER, [P1, P2], lenient(), n_samples=3,
                 fallback="number", sampler=canned(), on_record=seen.append)
        assert [r["unique_id"] for r in seen] == ["p1", "p2"]

    def test_a_failure_mid_loop_still_delivered_the_earlier_records(self):
        """The returned list is dropped when the loop raises, so hours of
        generation are lost unless the caller was handed each record."""
        seen = []

        def fails_on_the_second(model, tokenizer, problem, n):
            if problem == P2["problem"]:
                raise RuntimeError("out of memory")
            return list(SAMPLES[problem][:n])

        with pytest.raises(RuntimeError):
            evaluate(MODEL, TOKENIZER, [P1, P2], lenient(), n_samples=3,
                     fallback="number", sampler=fails_on_the_second,
                     on_record=seen.append)
        assert [r["unique_id"] for r in seen] == ["p1"]


class TestSeeding:
    """Before/after needs the sampling over the frozen subset repeatable."""

    @staticmethod
    def noisy(model, tokenizer, problem, n):
        draws = torch.randint(0, 10 ** 6, (n,)).tolist()
        return [Candidate(text=rf"\boxed{{{draw}}}", logprob_summary=-1.0)
                for draw in draws]

    def run(self, seed):
        results = evaluate(MODEL, TOKENIZER, [P1], lenient(), n_samples=3,
                           fallback="number", sampler=self.noisy, seed=seed)
        return results["records"][0]["candidates"]

    def test_the_same_seed_reproduces_the_candidates(self):
        assert self.run(7) == self.run(7)

    def test_a_different_seed_changes_them(self):
        assert self.run(7) != self.run(8)


class TestTokenLogprobs:
    """Realized-token logprobs without a full log_softmax copy."""

    def test_uniform_logits_give_log_one_over_vocab(self):
        got = token_logprobs_from_logits(torch.zeros(1, 2, 4),
                                         torch.tensor([[0, 3]]))
        assert got[0].tolist() == pytest.approx([-1.3863, -1.3863], abs=1e-4)

    def test_matches_log_softmax_then_gather(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 3, 7)
        ids = torch.randint(0, 7, (2, 3))
        want = torch.log_softmax(logits, -1).gather(
            -1, ids.unsqueeze(-1)).squeeze(-1)
        got = token_logprobs_from_logits(logits, ids)
        assert torch.allclose(got, want, atol=1e-6)


class TestEosIds:
    """A model with no stop token is a caller error, not a TypeError."""

    def test_no_configured_stop_token_raises_value_error(self):
        model = SimpleNamespace(
            generation_config=SimpleNamespace(eos_token_id=None))
        with pytest.raises(ValueError):
            _eos_ids(model, SimpleNamespace(eos_token_id=None))

    def test_a_list_of_stop_ids_is_kept_whole(self):
        model = SimpleNamespace(
            generation_config=SimpleNamespace(eos_token_id=[2, 3]))
        assert _eos_ids(model, SimpleNamespace(eos_token_id=3)) == (2, 3)


class TestOrder:
    """Records follow the order the problems came in."""

    def test_records_follow_problem_order(self):
        results = evaluate(MODEL, TOKENIZER, [P2, P1], lenient(), n_samples=3,
                           fallback="number", sampler=canned())
        assert [r["unique_id"] for r in results["records"]] == ["p2", "p1"]


class TestAccuracy:
    """Raw accuracy, and accuracy reweighted to MATH-500's level mix."""

    def test_raw_accuracy_is_the_fraction_of_correct_selections(self):
        results = evaluate(MODEL, TOKENIZER, [P1, P2], lenient(), n_samples=3,
                           fallback="number", sampler=canned())
        assert results["accuracy"] == pytest.approx(0.5)

    def test_level_reweighted_accuracy_uses_math500_level_shares(self):
        """MATH-500 holds 43 level-1 and 134 level-5 problems. With only
        those two levels present, the correct level-1 problem carries
        43 / (43 + 134) of the weight: 0.2429, against a raw 0.5."""
        results = evaluate(MODEL, TOKENIZER, [P1, P2], lenient(), n_samples=3,
                           fallback="number", sampler=canned())
        assert results["level_reweighted_accuracy"] == pytest.approx(
            43 / 177, abs=1e-4)

    def test_reweighting_is_the_identity_on_one_level(self):
        both_level_one = [P1, dict(P2, level=1)]
        results = evaluate(MODEL, TOKENIZER, both_level_one, lenient(),
                           n_samples=3, fallback="number", sampler=canned())
        assert results["level_reweighted_accuracy"] == pytest.approx(0.5)

    def test_no_problems_raise(self):
        with pytest.raises(ValueError):
            evaluate(MODEL, TOKENIZER, [], lenient(), sampler=canned())


class TestProblemValidation:
    """Bad problem fields fail before the sampling loop, not after it."""

    def test_unknown_level_raises_before_any_sampling(self):
        """The reweighting looks levels up in MATH-500's counts. Failing
        afterwards would discard a whole run of generation."""
        sampler = canned()
        with pytest.raises(ValueError):
            evaluate(MODEL, TOKENIZER, [dict(P1, level=6)], lenient(),
                     n_samples=3, sampler=sampler)
        assert sampler.calls == []

    def test_missing_field_raises_before_any_sampling(self):
        sampler = canned()
        problem = {k: v for k, v in P1.items() if k != "answer"}
        with pytest.raises(ValueError):
            evaluate(MODEL, TOKENIZER, [problem], lenient(), n_samples=3,
                     sampler=sampler)
        assert sampler.calls == []


class TestLogprobSummary:
    """The summary is the mean token logprob up to and including the
    first EOS. Positions after it hold -inf from the generation scores,
    so a multiplicative mask would turn the mean into NaN."""

    def test_no_eos_averages_every_token(self):
        logprobs = torch.tensor([[-1.0, -2.0, -3.0]])
        generated = torch.tensor([[5, 6, 7]])
        got = sequence_logprob_summary(logprobs, generated, eos_ids=0)
        assert got.tolist() == pytest.approx([-2.0])

    def test_eos_is_included_and_padding_after_it_is_not(self):
        logprobs = torch.tensor([[-1.0, -3.0, float("-inf"), float("-inf")]])
        generated = torch.tensor([[5, 0, 0, 0]])
        got = sequence_logprob_summary(logprobs, generated, eos_ids=0)
        assert got.tolist() == pytest.approx([-2.0])

    def test_any_of_several_eos_ids_ends_the_sequence(self):
        """Generation stops on the ids in the model's generation config,
        which on a chat checkpoint is a list. A row that stopped on one
        of them must not average in the -inf that follows."""
        logprobs = torch.tensor([[-1.0, -3.0, float("-inf")],
                                 [-1.0, -3.0, float("-inf")]])
        generated = torch.tensor([[5, 0, 0], [5, 1, 1]])
        got = sequence_logprob_summary(logprobs, generated, eos_ids=(0, 1))
        assert got.tolist() == pytest.approx([-2.0, -2.0])

    def test_rows_are_summarised_independently(self):
        logprobs = torch.tensor([[-1.0, -3.0, float("-inf")],
                                 [-2.0, -2.0, -2.0]])
        generated = torch.tensor([[5, 0, 0], [5, 6, 7]])
        got = sequence_logprob_summary(logprobs, generated, eos_ids=0)
        assert got.tolist() == pytest.approx([-2.0, -2.0])
