"""Contract tests for evaluate (role 3, and role 1 before training).

Written before the implementation, from the stub docstring and the
obligations docs/design-notes.md places on evaluate: build the material
bundle, store the per-problem record, select by argmax over the scorer,
check the all-rejected case first, grade with the caller's fallback, and
report raw and level-reweighted accuracy. Sampling is the one model-bound
step, so it is injected; scorers and the verifier are real.
"""
import pytest

from mini_reasoning.evaluate import evaluate
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

    def test_n_samples_bounds_the_stored_samples(self):
        results = evaluate(MODEL, TOKENIZER, [P1], lenient(), n_samples=2,
                           sampler=canned())
        assert len(results["records"][0]["samples"]) == 2


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
            {"text": r"so \boxed{42}", "logprob_summary": -1.0},
            {"text": "the answer is 7", "logprob_summary": -0.1},
            {"text": r"therefore \boxed{42}", "logprob_summary": -2.0},
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
