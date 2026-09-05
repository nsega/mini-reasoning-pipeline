"""Contract tests for the scorer interface (scaffold; the design is Naoki's).

The four properties recorded under "The contract tests" in
docs/design-notes.md, plus the role split that the third committed name
covers. Fakes are three-line classes: Protocol is structural, so nothing
needs importing to satisfy it.
"""
import pytest

from mini_reasoning.scorers import (
    REJECTED, Candidate, Composite, LogprobRank, ParseabilityGate,
    VoteAgreement,
)
from mini_reasoning.verifier import extract_final_candidate, verify_answer

CANDS = (
    Candidate(text=r"so \boxed{42}", logprob_summary=-1.0),
    Candidate(text="the answer is 7", logprob_summary=-0.1),
    Candidate(text=r"therefore \boxed{42}", logprob_summary=-2.0),
)


class RejectAll:
    def score(self, candidates):
        return [REJECTED] * len(candidates)


class Constant:
    def __init__(self, value):
        self.value = value

    def score(self, candidates):
        return [self.value] * len(candidates)


class TestOneCallContract:
    def test_scorers_share_one_call_contract(self):
        # No ground-truth parameter anywhere: selection runs where none exists.
        for scorer in (ParseabilityGate(), VoteAgreement(),
                       Composite((ParseabilityGate(), VoteAgreement()))):
            assert len(scorer.score(CANDS)) == len(CANDS)


class TestVeto:
    def test_parseability_gate_precedes_ranking(self):
        # Written against a stage built to break the rule: the veto must be a
        # property of the combination rule, not of stage good behaviour.
        composite = Composite((RejectAll(), Constant(float("inf"))))
        assert composite.score(CANDS) == [REJECTED] * len(CANDS)

    def test_all_rejected_is_data_not_an_error(self):
        # The deliberate counterpart to self_consistency_vote, which raises on
        # empty input because it is a selector and must decide.
        composite = Composite((RejectAll(),))
        assert composite.score(CANDS) == [REJECTED] * len(CANDS)


class TestPurity:
    def test_scoring_twice_is_equal_and_leaves_input_alone(self):
        # Load-bearing, not defensive: offline re-analysis of a composite
        # depends entirely on scorers being re-runnable over a stored record.
        composite = Composite((ParseabilityGate(), VoteAgreement()))
        before = list(CANDS)
        assert composite.score(CANDS) == composite.score(CANDS)
        assert list(CANDS) == before


class TestRoles:
    def test_same_verifier_serves_reward_and_validation_roles(self):
        # Roles differ only where extraction strictness applies.
        boxed = r"so \boxed{42}"
        assert (extract_final_candidate(boxed)
                == extract_final_candidate(boxed, fallback="number"))

        unboxed = "the answer is 7"
        assert verify_answer(extract_final_candidate(unboxed), "7") is False
        assert verify_answer(
            extract_final_candidate(unboxed, fallback="number"), "7")


class TestParseabilityGate:
    def test_rejects_what_no_answer_can_be_extracted_from(self):
        assert ParseabilityGate().score(CANDS) == [0.0, REJECTED, 0.0]

    def test_fallback_widens_what_survives_the_gate(self):
        gate = ParseabilityGate(fallback="number")
        assert gate.score(CANDS) == [0.0, 0.0, 0.0]


class TestLogprobRank:
    def test_ranks_by_summary(self):
        assert LogprobRank().score(CANDS) == [1.0, 2.0, 0.0]

    def test_ties_score_equally(self):
        tied = (
            Candidate(text="a", logprob_summary=-2.0),
            Candidate(text="b", logprob_summary=-2.0),
            Candidate(text="c", logprob_summary=-0.1),
        )
        assert LogprobRank().score(tied) == [0.0, 0.0, 2.0]

    def test_missing_summary_raises(self):
        with pytest.raises(ValueError):
            LogprobRank().score((Candidate(text=r"\boxed{1}"),))


class TestVoteAgreement:
    def test_counts_the_others_that_agree(self):
        assert VoteAgreement().score(CANDS) == [1.0, 0.0, 1.0]

    def test_unextractable_candidates_do_not_agree_with_each_other(self):
        unboxed = (
            Candidate(text="the answer is 7"),
            Candidate(text="the answer is 9"),
        )
        assert VoteAgreement().score(unboxed) == [0.0, 0.0]

    def test_fallback_widens_what_counts_as_an_answer(self):
        mixed = (
            Candidate(text="the answer is 7"),
            Candidate(text=r"\boxed{7}"),
        )
        assert VoteAgreement().score(mixed) == [0.0, 0.0]
        assert VoteAgreement(fallback="number").score(mixed) == [1.0, 1.0]


class TestComposite:
    def test_stages_sum(self):
        composite = Composite((Constant(1.0), Constant(2.0)))
        assert composite.score(CANDS) == [3.0] * len(CANDS)

    def test_composites_nest(self):
        inner = Composite((Constant(1.0), Constant(2.0)))
        assert Composite((inner, Constant(4.0))).score(CANDS) == [7.0] * 3

    def test_no_stages_scores_zero(self):
        assert Composite(()).score(CANDS) == [0.0] * len(CANDS)
