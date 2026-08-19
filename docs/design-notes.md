# Design notes: the scorer interface

> Status: draft — fills in as the interface design lands. This is the one
> design document that ships with the public repo. The README's Design
> rationale gives one bullet per decision; this file holds the full version:
> what was decided, what was rejected, and the evidence for each.
>
> Provenance: all decisions recorded here are self-written (the scorer
> interface is the capstone's closed-book constraint). The section skeleton
> was scaffolded; the content is Naoki's.

<!--
HOW TO FILL THIS FILE (delete this comment before merge)

Source: your filled answers in docs/scorer-interface-worksheet.md (branch-
only), rewritten in your own voice. Each section follows the same rubric:
decision / rejected option + the concrete failure it causes / evidence (a
lab measurement or a forcing call site) / cost accepted / the test that
pins it. Prefer stable references (function names) over file:line numbers,
which rot as implementations land. Worksheet question mapping is noted per
section as an HTML comment.
-->

## One interface or two

**Decision.**
scorers.py defines selection only. A scorer picks among sampled solutions and never receives the ground truth.
Verification stays in verifier.py as plain functions, and the caller (evaluate, or the GRPO trainer) runs them
whenever grading is needed.

**Rejected.**
Once combined interface that also grades (ground truth as an optional argument). Rejected because a selection scorer
that CAN see the ground truth can silently cheat at eval time and at real inference time there is no ground truth to give it anyway.

**Evidence.**
(1) All three stages of the lab's verdict. Parseability gate, rank, vote agreement. need no ground truth.
(2) Two of the four call sites (self-consistency voting, best-of-N selection) have no ground truth available at all.

**Cost accepted.**
Two swap points instead of one: Selection strategies swap in scorers.py, but reward strictness (boxed-only) swap at the trainer harness - the training loop calls the verifier with fallback= None: grpo.py itself never sees text or extraction.
The README's phrase "Scorer/reward is a swappable interface" overstates this, and I will fix it when the design lands.

**Pinned by.**
test_scorers_share_one_call_contract, every scorer is callable with samples alone. The call signature has no ground-truth parameter.

## What a scorer consumes and returns

<!-- worksheet Q1: signature, output type, and where the model asymmetry lives -->

**Decision.**
A scorer takes the whole candidate set and returns one score per candidate, positionally aligned with the input:
`score(candidates) -> list[float]`. It takes the set because the lab's own signals are relational — vote agreement scores
a candidate only bycomparison with the others, exactly as `self_consistency_vote` already takes the full list.
It returns scores rather than a winner because composition needs them: a gate that returned a winner would leave the ranking
stage nothing to rank. Selection is the caller's `argmax` over the returned scores.
The logprob asymmetry is place at the caller: every scorer receives the same precomputed material — the candidate text and,
when available, its logprob summary — so no scorer holds a model and every one of them is a pure function of its inpu.

**Rejected.**
A scorer that owns the model: evaluate already receives model and scorer as separate parameters, so this would put two model
references in one call, and it would make the pure heuristics untestable without a model double.
 A per-candidate signature(`score(candidate) -> float`): vote agreement cannot be expressed in it without giving each candidate
 a reference to the whole set it belongs to, which is circular. Returning a chosen index instead of scores: it makes
the composite's stages unable to feed each other.

**Evidence.**
(1) `self_consistency_vote(candidates)` already takes the whole set, because a majority is not computable one candidate at a
time —and vote agreement is the third stage of the lab's composition verdict.
(2) The composition verdict itself (parseability gate → rank → votagreement) requires each stage to hand the next a
per-candidate quantity, not a winner.
(3) Q3.0 removed GRPO from this protocol's consumers, so the reward path's scalar-per-rollout requirement does not constrain
this signature.

**Cost accepted.**
One shared input type that carries a field most scorers ignore, and the caller computes the logprob summary even in runs
where no scorer uses it.
A scorer that only needs one candidate still receives all of them, and every implementation must return a list
whose length matches its input — a contract the caller relies onpositionally.

**Pinned by.**
`test_scorers_share_one_call_contract` — every scorer,including a composite, is callable with the candidate set alone and
returns a list of the same length.

## Composition and the veto

<!-- worksheet Q2: is Composite a scorer; what "reject" means on each path -->

**Decision.** _TODO_

**Rejected.** _TODO_

**Evidence.** _TODO_

**Cost accepted.** _TODO_

**Pinned by.** _TODO_

## Three roles, one verifier

<!-- worksheet Q3: evaluation / reward / validation; where strictness binds -->

**Decision.**
All three roles share the same verifier: `verify_answer` is identical everywhere (sympy equivalence; None is always False), and so is the extraction machinery. The only per-role difference is extraction strictness.
Role 2 (reward) is strict boxed-only (`fallback=None`) — thereward's type sets the training regime, so this is not open. Roles 1 and 3 (baseline and validation) both use lenient extraction (a non-None fallback), and they must be identical to eachother: before/after is only meaningful when both evals grade with the same ruler. That equality is enforced by writting the same fallback argument at both eval call sites - the committed default already fails toward strict on omission.

**Rejected.**
All-strict evals: they would break comparability with officially graded numbers and leave the committed fallback contract (TestFallback) with no consumer in the pipeline.
A named-grader layer: one more indirection for a two-value choice.

**Evidence.**
(1) The no-fallback cost was measured precisely: 8/350 samples lost, worth 4 accuracy points at best-of-N under official grading —comparability with official numbers is why the fallback mode exists at all.
(2) The reward's type (binary, no fallback) sets the training regime, so role 2's strictness was settled in the lab. (3) The committed default already fails toward strict: omitting the argument means boxed-only (test_no_box_default_is_none).

**Cost accepted.**
Eval grading is more lenient than the signal that trained the model, so the circularity claim must be stated precisely. Validation reuses the same verify_answer and extraction machinery that produced the rewards. Only extraction leniency differs. The README will say it in those words.
role 1 == role 3 is held by discipline, not structure; the rule lives in this document and in a comment at both call sites.

**Pinned by.**
test_same_verifier_serves_reward_and_validation_roles — on aboxed sample, all three roles return the same verdict; on an unboxed sample, the reward grader returns None (False) while the eval grader resuse it - the roles differ exactly where the policy applies, and nowhere else.

## Forensics: obligation or convention

<!-- worksheet Q4: metadata from inside the scorer, or storage by the harness -->

**Decision.** _TODO_

**Rejected.** _TODO_

**Evidence.** _TODO_

**Cost accepted.** _TODO_

**Pinned by.** _TODO_

## The contract tests

<!-- worksheet Q5: 2-3 properties that must hold for ANY implementation -->

_TODO — one line per property, each naming the test in `tests/test_scorers.py`
that enforces it._
