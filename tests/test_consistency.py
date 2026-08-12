"""Contract tests for self-consistency voting (scaffold).

Week 1 criteria plus the None-exclusion policy the lab adopted on
evidence (up to 6/50 problems lost per n to None pooling).
"""
import pytest

from mini_reasoning.consistency import self_consistency_vote


class TestVoting:
    def test_majority(self):
        assert self_consistency_vote(["9", "9", "4"]) == "9"

    def test_deterministic_tiebreak(self):
        a = self_consistency_vote(["1", "2"])
        for _ in range(5):
            assert self_consistency_vote(["1", "2"]) == a

    def test_none_excluded_before_voting(self):
        # The policy change the lab paid 12 accuracy points to learn:
        # None must never out-vote a present correct answer.
        assert self_consistency_vote([None, "9", "14"]) in ("9", "14")
        assert self_consistency_vote([None, None, "9"]) == "9"

    def test_empty_input_is_loud(self):
        with pytest.raises(ValueError):
            self_consistency_vote([])
