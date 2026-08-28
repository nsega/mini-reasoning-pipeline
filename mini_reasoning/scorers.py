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
    """REJECTED for any candidate no answer can be extracted from, 0.0 otherwise."""

    def score(self, candidates: Sequence[Candidate]) -> list[float]:
        raise NotImplementedError


@dataclass(frozen=True)
class LogprobRank:
    """Ranks by the precomputed logprob summary. Requires it to be present."""

    def score(self, candidates: Sequence[Candidate]) -> list[float]:
        raise NotImplementedError


@dataclass(frozen=True)
class VoteAgreement:
    """How much each candidate's extracted answer agrees with the others.

    Distinct from consistency.self_consistency_vote, which returns a winner
    and owns its own None-exclusion policy; accepted duplication, recorded
    in the stub contract audit.
    """

    def score(self, candidates: Sequence[Candidate]) -> list[float]:
        raise NotImplementedError


@dataclass(frozen=True)
class Composite:
    """Stages in order, scores summed. Satisfies Scorer, so composites nest."""
    stages: tuple[Scorer, ...]

    def score(self, candidates: Sequence[Candidate]) -> list[float]:
        raise NotImplementedError
