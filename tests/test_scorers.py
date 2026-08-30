"""Contract tests for the scorer interface (scaffold; the design is Naoki's).

The four properties recorded under "The contract tests" in
docs/design-notes.md, plus the role split that the third committed name
covers. Fakes are three-line classes: Protocol is structural, so nothing
needs importing to satisfy it.
"""
import pytest

from mini_reasoning.scorers import (
    REJECTED, Candidate, Composite, ParseabilityGate, VoteAgreement,
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
