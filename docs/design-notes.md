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

**Decision.**
A scorer takes the whole candidate set and returns one score per candidate, positionally aligned with the input:
`score(candidates) -> list[float]`. It takes the set because the lab's own signals are relational — vote agreement scores
a candidate only bycomparison with the others, exactly as `self_consistency_vote` already takes the full list.
It returns scores rather than a winner because composition needs them: a gate that returned a winner would leave the ranking
stage nothing to rank. Selection is the caller's `argmax` over the returned scores.
The logprob asymmetry is place at the caller: every scorer receives the same precomputed material — the candidate text and,
when available, its logprob summary — so no scorer holds a model and every one of them is a pure function of its iput.

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

**Decision.**
`Composite` satisfies the same protocol as any other scorer —`score(candidates) -> list[float]` — so composites nest inside composites and every call site depends on one type. Stages run in the lab's order
(parseability gate, then rank, then vote agreement) and their scores sum.
A gate expresses rejection as `-inf` for that candidate, which is an absorbing element: no later stage can lift it, whatever it returns, so the caller's `argmax` can never land on a rejected candidate. The veto is
enforced by `Composite`'s combination rule, not by each stage behaving well. When every candidate is rejected, the composite returns a list of all -inf and says nothing more; detecting that no candidate is selectable is the caller's job, because a scorer scores and does not decide.

**Rejected.**
Dropping rejected candidates from the list instead of marking them: it breaks the positional contract from Q1 — the returned list wouldno longer align with the input the caller holds — and it hides the
rejection from any later inspection. A separate filter phase running before the scorers: it would put gates outside the protocol, so a gate could not itself be a composite of gates. Trusting each stage to leave rejected candidates alone: one stage with a different sign convention silently breaks the veto, and nothing turns red.
Rasing from inside the composite: a scorer that raises cannot be composed with one that does not, and the same all-rejected list is a legitimate intermediate state inside a larger composite.

**Evidence.**
(1) Q1 fixed the positional contract — same length in, samelength out — so removal is not available to a stage.
(2) `-inf` isabsorbing under the sum, which makes the veto a property of the combination rule rather than of stage behaviour.
(3) The lab's verdict is an ordered composition (gate, then rank, then vote agreement), so the gate's outcome must survive two later stages to mean anything.
(4) The group-size hazard that made masking necessary on the reward path no longer applies here — Q3.0 moved GRPO outside this protocol — so the reason formasking is now the positional contract alone.

**Cost accepted.**
Scores are not a bounded scale: they carry `-inf` as a sentinel, so any implementation combining them must be sum-compatible, and a stage returning a large positive number cannot be reasoned about independently of the gate.
Every selection call site must check for the all-rejected case before taking argmax, or it will silently select the first candidate; this document and a comment at each call site carry that rule.

**Pinned by.**
`test_parseability_gate_precedes_ranking` — a composite of agate that rejects a candidate and an adversarial later stage that returns `+inf` for everything still scores that candidate `-inf`; the gate's veto survives a stage built to break it.

## Three roles, one verifier

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

**Decision.**
Forensics is a harness responsibility, not an interface obligation: the protocol stays `score(candidates) -> list[float]` and nothing more. `evaluate` stores the per-problem record it already promises — samples, extracted candidates, scores, and the selected index — and tharecord answers the questions the lab actually had to ask. The one fact itcannot answer is per-stage attribution inside a composite, because summing stage scores discards the breakdown; hat breakdown is accepted as lost, and if it is ever needed the composite can be re-run offline over the stored candidates, because every scorer is a pure function of its input (Q1). Provided that record holds the precomputed material, including the logprob summary, and not just the raw text. That requirement lands on `evaluate`'s record, not on the protocol.

**Rejected.**
Returning `(score, metadata)` from every scorer: it forces every call site, including the composite's inner loop, to unpack metadata it discards, and it makes the positional-alignment contract from Q1 harder to state. Requiring an `explain()` method on the protocol: every implementation, including a three-line test fake, would have to write one for a question the lab has never yet needed to ask. A debug method on Composite alone: it would be a second way to call a scorer that only one class supports, and re-running the pure composite offline already recovers the same numbers.

**Evidence.**
(1) The recoverability test: with candidates and final scores stored, "why was this candidate not selected" is answerable — a`-inf` identifies a gate rejection — so the interface does not need tocarry that.
(2) The lab's selected-None-rate discovery needed only raw stored candidates, which is a harness capability, not a scorer one.
(3) What storage cannot recover is a stage-level breakdown: the sum in Q2 is a lossy combination, and the parts exist only inside `Composite`.
(4) `tests/test_scorers.py` committed three placeholder names covering the call contract, the veto, and the three roles — and none for forensics.

**Cost accepted.**
A composite's score is opaque about which stage produced it, so a future question of the form "did rank or vote agreement cost this candidate the selection" cannot be answered from stored records alone answering it means re-running the composite over the stored candidates rather than reading it off the record.

**Pinned by.**
Nothing in `tests/test_scorers.py` — deliberately. Thisdecision is pinned negatively: the scorer protocol has exactly one method, so `test_scorers_share_one_call_contract` passing with a three-line fake that implements only `score` is itself the evidence that forensics was not made an obligation.

## The contract tests

<!-- worksheet Q5: 2-3 properties that must hold for ANY implementation -->

_TODO — one line per property, each naming the test in `tests/test_scorers.py`
that enforces it._
