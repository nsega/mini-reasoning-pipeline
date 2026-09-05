"""The swappable scorer interface: selection only.

Selection ranks sampled solutions against each other and never sees the
ground truth; verification lives in verifier.py and is run by the caller.
The full record — rejected options, evidence, costs — is in
docs/design-notes.md.

- A scorer takes the whole candidate set and returns one score per
  candidate, positionally aligned with the input. Vote agreement is
  relational, so a per-candidate signature cannot express it, and
  composition needs scores rather than a winner. Selection is the
  caller's argmax.
- Every scorer is a pure function of precomputed material: the caller
  builds it, so no scorer holds a model and none needs a model double
  in tests.
- Composite satisfies this same protocol and nests inside itself. Stages
  sum, and a gate rejects by scoring -inf, an absorbing element. The veto
  is a property of the combination rule, not of stage behaviour.
- An all-rejected set returns all -inf and does not raise: a scorer
  scores, it does not decide. Callers check before taking argmax.
- Nothing here reports why. Forensics is the harness's job; purity is
  what lets a composite be re-run offline over the stored record.
"""
from dataclasses import dataclass
from typing import Protocol, Sequence

from mini_reasoning import verifier

REJECTED = float("-inf")


@dataclass(frozen=True)
class Candidate:
    """The precomputed material every scorer receives.

    Built once by the caller, which is what keeps scorers pure. Fields a
    given scorer does not use are simply ignored by it.
    """
    text: str
    logprob_summary: float | None = None   # None when not computed


class Scorer(Protocol):
    def score(self, candidates: Sequence[Candidate]) -> list[float]:
        """One score per candidate, positionally aligned with the input.

        Must be pure: equal input gives equal output, and `candidates` is
        not mutated. REJECTED marks a candidate no later stage may lift.
        """
        ...


@dataclass(frozen=True)
class ParseabilityGate:
    """REJECTED where no answer can be extracted, 0.0 everywhere else."""
    fallback: str | None = None    # boxed-only unless the caller says otherwise

    def score(self, candidates: Sequence[Candidate]) -> list[float]:
        return [
            REJECTED
            if verifier.extract_final_candidate(c.text, self.fallback) is None
            else 0.0
            for c in candidates
        ]


@dataclass(frozen=True)
class LogprobRank:
    """Ranks by the precomputed logprob summary. Requires it to be present.

    A candidate scores the number of candidates strictly below it, so the
    scale is bounded by the set size and stays commensurate with vote
    agreement under the composite's sum. Equal summaries score equally,
    since nothing distinguishes them but position.
    """

    def score(self, candidates: Sequence[Candidate]) -> list[float]:
        summaries = [c.logprob_summary for c in candidates]
        if any(s is None for s in summaries):
            raise ValueError(
                "LogprobRank needs a logprob summary for every candidate; "
                "the caller computes it when building the material")
        return [float(sum(other < s for other in summaries))
                for s in summaries]


@dataclass(frozen=True)
class VoteAgreement:
    """How much each candidate's extracted answer agrees with the others.

    Distinct from consistency.self_consistency_vote, which returns a winner
    and owns its own None-exclusion policy; accepted duplication, recorded
    in the stub contract audit.
    """
    fallback: str | None = None    # boxed-only unless the caller says otherwise

    def score(self, candidates: Sequence[Candidate]) -> list[float]:
        answers = [verifier.extract_final_candidate(c.text, self.fallback)
                   for c in candidates]
        # None agrees with nothing: pooling failed extractions on one key is
        # the failure self_consistency_vote excludes them to avoid.
        return [0.0 if a is None else float(answers.count(a) - 1)
                for a in answers]


@dataclass(frozen=True)
class Composite:
    """Stages in order, scores summed. Satisfies Scorer, so composites nest."""
    stages: tuple[Scorer, ...]

    def score(self, candidates: Sequence[Candidate]) -> list[float]:
        totals = [0.0] * len(candidates)
        for stage in self.stages:
            for i, value in enumerate(stage.score(candidates)):
                # REJECTED is absorbing by rule, not by arithmetic: summing
                # it with a stage returning +inf would give nan.
                if totals[i] == REJECTED or value == REJECTED:
                    totals[i] = REJECTED
                else:
                    totals[i] += value
        return totals
