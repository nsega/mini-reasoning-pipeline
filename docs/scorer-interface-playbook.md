# Scorer interface — the signature-stop playbook

> Provenance: Claude-written process scaffolding. This file contains **no
> design decisions** — only the mechanics of the design phase: where exactly
> to stop, how to check you stopped there, and what to do with what you
> rejected. Companions: `scorer-interface-worksheet.md` (what to decide),
> `scorer-interface-illustration.md` (what a finished answer looks like).
> Code samples below stay in the illustration's analogue domain (the
> dependency resolver's `VersionPicker`), never the scorer. Delete this file
> once the design lands.

## Practice 1 — stop at signatures

**The boundary rule: design is what the type checker sees. Implementation is
what only the runtime sees.**

In bounds — these ARE design, write them fully:

- `Protocol` classes with `...` bodies. For a protocol, `...` is the
  *finished* state, not a placeholder — protocols never gain bodies.
- Frozen dataclass field lists, including defaults.
- Enum members.
- Constructor and function signatures **including default values**. The
  fail-toward-strict direction is expressible entirely at this layer — your
  own `extract_final_candidate(text, fallback=None)` (`verifier.py:22`)
  already proves it: the safety property lives in the default, not in any
  body.
- Docstrings. The module docstring is where the decision ledger lands
  (Practice 2).

Out of bounds — implementation, defer all of it:

- Any control flow: `if`, `for`, `while`, `try`.
- Any expression that does work: a `Counter`, a `max`, arithmetic.
- Concrete method bodies beyond `raise NotImplementedError`.

Borderline eliminated by construction: prefer frozen dataclasses for concrete
classes, and there is no `__init__` body left to police.

### The landed shape (analogue domain)

```python
"""Version picking. Decisions: <ledger lives here — see Practice 2>."""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, Sequence


class AdmissionPolicy(Enum):
    STRICT = auto()
    LENIENT = auto()


@dataclass(frozen=True)
class Candidate:
    version: str
    requested_range: str
    registry_meta: dict | None = None   # None when unfetched: offline mode
    lock_pin: str | None = None


class Picker(Protocol):
    def score(self, cands: Sequence[Candidate]) -> list[float]:
        """Positionally aligned with cands; -inf marks inadmissible."""
        ...


@dataclass(frozen=True)
class SemverPicker:
    policy: AdmissionPolicy = AdmissionPolicy.STRICT   # omission fails strict

    def score(self, cands):
        raise NotImplementedError


@dataclass(frozen=True)
class Composite:
    stages: tuple[Picker, ...]

    def score(self, cands):
        raise NotImplementedError
```

About forty lines, and every one of them is a decision. That is the whole
point: the review reads only decisions, and the closed-book rule stays intact
because no algorithmic content exists yet to have been sourced from anywhere.

### Checks that you stopped in the right place

- `python -c "from mini_reasoning import scorers"` succeeds.
- Every concrete `def` body is exactly `raise NotImplementedError`.
- Nothing in the file could fail a test except by `NotImplementedError`.

## Practice 1½ — tests against signatures (the step between)

"Tests land together with the design" is literal, through two mechanisms.

**(a) In-test fakes.** `Protocol` is structural: a three-line class in the
test file satisfies it, no inheritance, no imports beyond the interface. Fakes
prove the interface is implementable and give composition tests something to
compose before any real implementation exists.

```python
class RejectAll:
    def score(self, cands): return [float("-inf")] * len(cands)

class Constant:
    def __init__(self, v): self.v = v
    def score(self, cands): return [self.v] * len(cands)
```

(If you want `isinstance` checks in the shared-contract test, mark the
protocol `@runtime_checkable` — knowing it checks method *names*, not
signatures. That choice is itself a design-layer decision; record it.)

**(b) Failure-type triage.** With signatures landed and the three
placeholders in `tests/test_scorers.py` converted to real tests, run
`pytest tests/test_scorers.py` and read failures by *type*:

| Failure type | Meaning | Action |
|---|---|---|
| ImportError / collection error | file is broken | fix now |
| TypeError / AttributeError raised by a test calling the interface | the design cannot express its own test — a design bug found **before implementing** | revise the signature |
| assertion failure in a fakes-only test | the property is wrong, or the fake is | decide which, record it |
| NotImplementedError | the correct end-state of the design phase | stop — implementation is a later commit |

**"Red for the right reason" is the exit criterion**: the design phase is done
when every failure in `test_scorers.py` is a `NotImplementedError`. The
TypeError row is what this practice buys — it is the cheapest design review
that exists, and it runs before a single body is written.

Convert the placeholders one at a time, as their question gets answered
(Q1 → `test_scorers_share_one_call_contract`, Q2 →
`test_parseability_gate_precedes_ranking`, Q3 →
`test_same_verifier_serves_reward_and_validation_roles`), then add the Q5
property tests. The three names at `tests/test_scorers.py:12,17,22` are
committed spec — keep them.

## Practice 2 — the rejected option, and where it lives

One decision has three homes, in lifecycle order:

1. **Worksheet blanks** (drafting): the full 2–4 bullets per question, in the
   rubric shape — decision / rejected option + the failure it causes / cost
   accepted / the assert.
2. **`scorers.py` module docstring** (the landed ledger): the stub's four
   *questions* are replaced by four *answers*, in place. The questions do not
   outlive their answers; the file documents itself.
3. **README Design rationale** (the distillation): one bullet per decision,
   in the house voice.

The house voice, read off `README.md:28-42`: **bold decision, mechanism, then
the evidence in one clause**. Every existing bullet's evidence is either a lab
measurement (8/350, 6/50-per-n, 12 points, 40 steps) or a forcing constraint.
Yours will be the same: a lab number, or a forcing call site
(`grpo.py:21` needs scalars-per-group; `consistency.py:17` has no ground
truth). Name it. "For flexibility" and "for cleanliness" are not evidence.

### The compression, demonstrated (analogue)

Worksheet answer: the four Q3 bullets from the illustration. README bullet:

```
- **Admission policy binds at Picker construction**: one policy per whole
  resolution is what makes lockfiles reproducible; the per-call flag was
  rejected because one forgotten argument at one of six recursive call sites
  yields a lockfile that differs on someone else's machine. STRICT is the
  constructor default, so omission fails toward reproducibility. Cost:
  switching policy mid-run means constructing a second picker.
```

Same content, one breath: decision, rejected option, failure, default
direction, cost. A reviewer can polish the voice without re-deriving the
substance.

## Practice 3 — the stub-contract audit (run last, expect a bite)

The design must coexist with every contract already committed to this repo.
Fill the verdict column after Q1–Q5 are answered, not before.

| # | Committed contract | Where | Verdict |
|---|---|---|---|
| 1 | `extract_final_candidate(text, fallback=None)`, boxed-only default | `verifier.py:22`, `test_verifier.py:23-25` | |
| 2 | Some opt-in fallback exists and rescues a bare number | `test_verifier.py:29-34` | |
| 3 | `verify_answer(candidate, ground_truth)`; None is always False | `verifier.py:26`, `test_verifier.py:44-45` | |
| 4 | `self_consistency_vote(candidates)`: bare list in, winner out; None excluded pre-vote; empty raises; deterministic tiebreak | `consistency.py:17`, `test_consistency.py` | |
| 5 | `group_relative_advantages(rewards)`: group ≥ 2 or ValueError; zero-variance → exact zeros | `grpo.py:21`, `test_grpo.py:23-31` | |
| 6 | `grpo_loss(logprobs, advantages)`: descent direction, advantages detached | `grpo.py:25`, `test_grpo.py:34-45` | |
| 7 | `evaluate(model, tokenizer, problems, scorer)`: model and scorer are separate parameters; per-problem records stored | `evaluate.py:12-17` | |
| 8 | Three test names in `test_scorers.py` | `tests/test_scorers.py:12,17,22` | |
| 9 | `--n-samples` default 7 (the reward path's group size) | `run_pipeline.py:17-18` | |

Verdict rules:

- **UNCHANGED**: name, in one phrase, the design element that satisfies it.
  An unexamined checkmark is not a verdict.
- **CHANGED**: this is a *finding*, not a failure. Record which side was
  wrong — the design (go revise it) or the stub (say what assumption the stub
  made that did not survive, and why). A stub-was-wrong finding is
  README-worthy.
- **All nine UNCHANGED with no tension found anywhere: be suspicious.** At
  minimum, the relation between row 4 (`self_consistency_vote` takes a bare
  candidate list and returns a winner) and the README's "composition is the
  interface" verdict (`README.md:28-34`) has to be *stated* somewhere: is the
  vote itself a scorer, a component that a vote-agreement scorer wraps, or
  outside the interface entirely? Any of those can be right. No answer means
  the audit did not look.

## The commit shape

The design phase is **one commit** touching exactly three files:

- `mini_reasoning/scorers.py` — ledger docstring + signatures
- `tests/test_scorers.py` — three real tests + Q5 properties, red only by
  `NotImplementedError`
- `README.md` — the new bullets under Design rationale

If this commit wants to touch `verifier.py`, `consistency.py`, `grpo.py`, or
their tests, that is a Practice-3 finding escaping into the diff — stop and
record it first, then decide.

Implementation commits follow separately, module by module, turning red to
green **with no further edits to the tests**. A test edit during
implementation is a design change wearing a disguise; send it back through
the audit.
