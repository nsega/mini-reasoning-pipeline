# Design notes: the scorer interface

> Status: the interface design is settled; further sections land as
> implementation decisions are made. This is the one design document that
> ships with the public repo. The README's Design rationale gives one bullet
> per decision; this file holds the full version: what was decided, what was
> rejected, and the evidence for each.
>
> Provenance: the scorer interface is the capstone's closed-book
> constraint, so every decision recorded here is self-written with three
> exceptions. The section skeleton was scaffolded; the content is Naoki's.
> The exceptions are *The fallback contract*, drafted with Claude while the
> verifier was implemented, and the two `grpo.py` sections, drafted with
> Claude while that module was implemented, each from options Claude
> proposed and Naoki accepted.

## One interface or two

**Decision.**
scorers.py defines selection only. A scorer picks among sampled solutions and never receives the ground truth.
Verification stays in verifier.py as plain functions, and the caller (evaluate, or the GRPO trainer) runs them
whenever grading is needed.

**Rejected.**
One combined interface that also grades (ground truth as an optional argument). Rejected because a selection scorer
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
a candidate only by comparison with the others, exactly as `self_consistency_vote` already takes the full list.
It returns scores rather than a winner because composition needs them: a gate that returned a winner would leave the ranking
stage nothing to rank. Selection is the caller's `argmax` over the returned scores.
The logprob asymmetry is placed at the caller: every scorer receives the same precomputed material — the candidate text and,
when available, its logprob summary — so no scorer holds a model and every one of them is a pure function of its input.

Extraction is not part of the material. The parseability gate and vote agreement each call `extract_final_candidate` themselves, because extraction needs no model and so creates none of the asymmetry the bundle exists to remove. Each carries its own `fallback` as a construction-time field defaulting to boxed-only, so the caller still chooses strictness and an omission still fails toward strict. On the selection path the caller passes the eval fallback: a gate stricter than the grader that follows it would discard candidates the grader could have scored, which is the 8/350 samples the fallback exists to recover.

**Rejected.**
A scorer that owns the model: evaluate already receives model and scorer as separate parameters, so this would put two model
references in one call, and it would make the pure heuristics untestable without a model double.
A per-candidate signature (`score(candidate) -> float`): vote agreement cannot be expressed in it without giving each candidate
 a reference to the whole set it belongs to, which is circular. Returning a chosen index instead of scores: it makes
the composite's stages unable to feed each other.

**Evidence.**
(1) `self_consistency_vote(candidates)` already takes the whole set, because a majority is not computable one candidate at a
time —and vote agreement is the third stage of the lab's composition verdict.
(2) The composition verdict itself (parseability gate → rank → vote agreement) requires each stage to hand the next a
per-candidate quantity, not a winner.
(3) Keeping verification outside this protocol ([One interface or two](#one-interface-or-two)) removed GRPO from this protocol's consumers, so the reward path's scalar-per-rollout requirement does not constrain
this signature.

**Cost accepted.**
One shared input type that carries a field most scorers ignore, and the caller computes the logprob summary even in runs
where no scorer uses it.
A scorer that only needs one candidate still receives all of them, and every implementation must return a list
whose length matches its input — a contract the caller relies on positionally.

Extraction runs inside two stages rather than once at the caller, so the strictness rule now lives at three call sites instead of two, and a stage reused in a stricter context would need its fallback set again.

**Pinned by.**
`test_scorers_share_one_call_contract` — every scorer, including a composite, is callable with the candidate set alone and
returns a list of the same length.

## Composition and the veto

**Decision.**
`Composite` satisfies the same protocol as any other scorer —`score(candidates) -> list[float]` — so composites nest inside composites and every call site depends on one type. Stages run in the lab's order
(parseability gate, then rank, then vote agreement) and their scores sum.
A gate expresses rejection as `-inf` for that candidate, which is an absorbing element: no later stage can lift it, whatever it returns, so the caller's `argmax` can never land on a rejected candidate. The veto is
enforced by `Composite`'s combination rule, not by each stage behaving well. When every candidate is rejected, the composite returns a list of all -inf and says nothing more; detecting that no candidate is selectable is the caller's job, because a scorer scores and does not decide.

**Rejected.**
Dropping rejected candidates from the list instead of marking them: it breaks the positional contract — the returned list would no longer align with the input the caller holds — and it hides the
rejection from any later inspection. A separate filter phase running before the scorers: it would put gates outside the protocol, so a gate could not itself be a composite of gates. Trusting each stage to leave rejected candidates alone: one stage with a different sign convention silently breaks the veto, and nothing turns red.
Raising from inside the composite: a scorer that raises cannot be composed with one that does not, and the same all-rejected list is a legitimate intermediate state inside a larger composite.

**Evidence.**
(1) The signature fixed the positional contract — same length in, same length out — so removal is not available to a stage.
(2) `-inf` is absorbing under the sum, which makes the veto a property of the combination rule rather than of stage behaviour.
(3) The lab's verdict is an ordered composition (gate, then rank, then vote agreement), so the gate's outcome must survive two later stages to mean anything.
(4) The group-size hazard that made masking necessary on the reward path no longer applies here, because GRPO is not a consumer of this protocol ([One interface or two](#one-interface-or-two)), so the reason for masking is now the positional contract alone.

**Cost accepted.**
Scores are not a bounded scale: they carry `-inf` as a sentinel, so any implementation combining them must be sum-compatible, and a stage returning a large positive number cannot be reasoned about independently of the gate.
Every selection call site must check for the all-rejected case before taking argmax, or it will silently select the first candidate; this document and a comment at each call site carry that rule.

The None filter runs twice — once inside `self_consistency_vote` by module policy, once as the composite's first stage, and that redundancy is accepted: the two owners answer different questions, one protecting a vote it does not control the input of, the other producing a score. Neither can assume the other ran.

**Pinned by.**
`test_parseability_gate_precedes_ranking` — a composite of a gate that rejects a candidate and an adversarial later stage that returns `+inf` for everything still scores that candidate `-inf`; the gate's veto survives a stage built to break it.

## What LogprobRank returns

**Decision.**
`LogprobRank` scores each candidate with the number of candidates strictly below it by logprob summary: 0 for the lowest, `n - 1` for the highest when nothing ties, and equal scores for equal summaries. A missing summary raises `ValueError` rather than being read as a low rank.

**Rejected.**
Returning the raw logprob summary as the score. It preserves how far apart candidates are, but the values are unbounded below, so a candidate at -50 swamps any vote-agreement count under the composite's sum and the stage cannot be reasoned about beside its peers. [Composition and the veto](#composition-and-the-veto) already accepts that cost in the positive direction ("a stage returning a large positive number cannot be reasoned about independently of the gate"); this is the same problem mirrored.
Min-max normalising the summaries into [0, 1]. Bounded, and it keeps relative spacing, but an all-equal set divides by zero and a single degenerate outlier flattens every other candidate into a narrow band.
Scoring a missing summary as the lowest rank. It would let a run that never computed the summaries produce a plausible-looking ranking, which is the silent failure the loud-error policy exists to prevent.

**Evidence.**
(1) The stage exists to be composed. The lab's verdict is gate, then rank, then vote agreement, so this stage's output is summed with a vote-agreement count bounded by the set size; commensurate scales are what make that sum mean anything.
(2) Ties scoring equally follows from the positional contract. Two candidates with identical summaries differ only in where they sit in the list, so letting position decide would make the caller's argmax prefer an earlier candidate for no reason. This is the opposite of `self_consistency_vote`, which must break ties because it returns a winner rather than a score.
(3) The class docstring already required the summary to be present; the raise is what makes that enforceable rather than aspirational, and `evaluate` is the one place that builds it (audit row 7).

**Cost accepted.**
The size of the gap is discarded: a candidate far ahead on logprob scores exactly one more than the runner-up. That is deliberate rather than merely tolerated, because the lab measured per-token logprob scoring over-selecting degenerate fluency, so a wide logprob margin is not evidence this design wants to amplify.

**Pinned by.**
`TestLogprobRank` in `tests/test_scorers.py`: ranking by summary, ties scoring equally, and a missing summary raising. Nothing pinned this stage before, it being the one scorer the committed contract tests never reached.

## Three roles, one verifier

**Decision.**
All three roles share the same verifier: `verify_answer` is identical everywhere (sympy equivalence; None is always False), and so is the extraction machinery. The only per-role difference is extraction strictness.
Role 2 (reward) is strict boxed-only (`fallback=None`) — the reward's type sets the training regime, so this is not open. Roles 1 and 3 (baseline and validation) both use lenient extraction (a non-None fallback), and they must be identical to each other: before/after is only meaningful when both evals grade with the same ruler. That equality is enforced by writing the same fallback argument at both eval call sites - the committed default already fails toward strict on omission.

**Rejected.**
All-strict evals: they would break comparability with officially graded numbers and leave the committed fallback contract (Test Fallback) with no consumer in the pipeline.
A named-grader layer: one more indirection for a two-value choice.

**Evidence.**
(1) The no-fallback cost was measured precisely: 8/350 samples lost, worth 4 accuracy points at best-of-N under official grading —comparability with official numbers is why the fallback mode exists at all.
(2) The reward's type (binary, no fallback) sets the training regime, so role 2's strictness was settled in the lab. (3) The committed default already fails toward strict: omitting the argument means boxed-only (test_no_box_default_is_none).

**Cost accepted.**
Eval grading is more lenient than the signal that trained the model, so the circularity claim must be stated precisely. Validation reuses the same verify_answer and extraction machinery that produced the rewards. Only extraction leniency differs. The README will say it in those words.
role 1 == role 3 is held by discipline, not structure; the rule lives in this document and in a comment at both call sites.

**Pinned by.**
test_same_verifier_serves_reward_and_validation_roles — on a boxed sample, all three roles return the same verdict; on an unboxed sample, the reward grader returns None (False) while the eval grader rescues it - the roles differ exactly where the policy applies, and nowhere else.

## The fallback contract

**Decision.**
`extract_final_candidate(text, fallback=None)` takes exactly one lenient mode, `"number"`, alongside the boxed-only default. The rescue is a second strategy rather than a competing one: it runs only where the strict rule found nothing, so a boxed answer always wins. All three ways of failing the strict rule (no `\boxed` at all, no brace after it, and a brace the sample ends before closing) collapse to a single exit, so the `fallback` argument alone decides whether a sample without a usable box has an answer at all. `"number"` takes the last number in the text, consistent with the last box winning on the strict path, and strips thousands separators. An empty `\boxed{}` counts as no answer and reaches the rescue rather than returning `""`. An unrecognised mode raises `ValueError` before the text is searched.

**Rejected.**
The official `number_then_full` chain, whose second link falls back to the whole sample text. Its second link can only match when the entire sample is itself a bare answer, which never happens for chain-of-thought output, so it recovers nothing measurable while handing the parser a paragraph that implicit multiplication turns into a product of symbols. The 8/350 recovery the mode exists for comes from the number link alone.
Distinguishing the three strict-rule failures, so that a truncated `\boxed{` would be treated differently from a sample that never boxed at all. The difference is not visible to the grader, and it would make the rescue fire for one cut-off sample and not another.
Silently degrading an unknown mode to boxed-only. It makes a typo behave correctly on most samples and fail rarely, which is the worst available failure shape given that strictness now lives at three call sites.

**Evidence.**
(1) A truncated box argues for the funnel rather than against it: `rfind` targets the last `\boxed`, so `\boxed{1} then \boxed{2` extracts as nothing under the strict rule even though the sample did box an answer, and the rescue recovers the model's final attempt.
(2) The mode check must precede the search. A misspelled mode that only raised where no number was found would pass silently on most samples and leave roles 1 and 3 grading with different rulers, which *Three roles, one verifier* already records as held by discipline rather than structure.
(3) sympy parses `"1,000"` into the tuple `(1, 0)` rather than raising, so an unstripped separator surfaces as a TypeError inside grading rather than as a wrong answer.

**Cost accepted.**
The mode is a bare string rather than an enum, so the three call sites can still disagree. The `ValueError` turns a typo from a silent divergence into a loud one, but nothing stops a site from being left on the default and quietly grading stricter than its peers. Two behaviours are also unpinned by any test: the thousands-separator stripping, and the empty-box decision.

**Pinned by.**
`TestFallbackContract` in `tests/test_verifier.py`: a boxed answer wins over the rescue, no number stays None, the last number wins, a sign and a decimal point survive, a truncated box is rescued under a mode and stays None under the default, an empty box is not an answer, and an unknown mode raises on every input. The committed `test_fallback_rescues_bare_number` continues to pin only that some non-None mode exists.

## What group_relative_advantages calls flat

**Decision.**
A group is flat when every reward in it equals the first, compared exactly on the rewards themselves, and a flat group returns exact zeros. Every other group returns `(r - mean) / (std + eps)` with the unbiased std along the last axis, which is the group axis. A group axis shorter than two raises `ValueError`, because the n-1 divisor makes a group of one NaN rather than zero. Every input is normalised in float32 whatever its dtype: `verify_answer` returns `bool`, so the tensor a harness builds straight from its verdicts is `torch.bool`, on which `torch.mean` raises, and float16 cannot represent the epsilon at all, so a near-flat half group would divide by zero. A NaN or infinite reward, an empty batch, and a 0-dim tensor each raise `ValueError` rather than pass through: NaN is never equal to itself, so a group holding one is never flat and would come out all-NaN, and the loss's global mean would then spread that NaN across every row's gradient.

**Rejected.**
Judging flatness from the std being zero. Seven copies of 0.1 do not sum to 0.7 in float32, so their std is near 1e-8 rather than 0, and the epsilon then turns rounding noise into an advantage of -0.41 on every rollout. Seven is the default `--n-samples`, and the group sizes the committed tests use (3 and 4) happen not to show it.
Judging flatness by a tolerance on the std. It replaces one unmeasured constant with another, and a narrow but real spread of rewards under the tolerance would be zeroed with nothing turning red.
Raising on a flat group. The trainer steps through them, and with a binary reward they are the common case: a problem the policy gets entirely wrong or entirely right.

**Evidence.**
(1) The probe that found it: constant groups of 0.1, 0.3, 0.7 and 1/3 all have a nonzero float32 std at n = 7 and an exactly zero one at n = 3 and n = 4, so the hazard is invisible at the group sizes the committed tests use and present at the default.
(2) What a wrong answer does to training. Advantages of -0.41 on every rollout are not noise around zero but a uniform push down on every logprob in the group, so each all-same-reward problem would lower the policy's probability on its own samples. With no KL term (the no-KL default the module docstring records), nothing else acts on that step, so the exact zero is the whole of what keeps it inert.
(3) The reward role's type is binary ([Three roles, one verifier](#three-roles-one-verifier)), and 0 and 1 are exact in float32, so the pipeline as committed never triggers this. A shaped or partial-credit reward would, which is the swap the trainer harness owns.

**Cost accepted.**
Exact comparison means a group whose rewards differ only by floating-point noise is not flat and is normalised: two rewards that differ at 1e-7 produce advantages of order one. Binary and rational rewards never do this. A reward computed through floating-point arithmetic would need rounding by the harness before it is grouped, and this document is where that rule lives.

**Pinned by.**
`test_flat_group_of_inexact_floats_is_still_exact_zeros` in `tests/test_grpo.py`, written against the std-based check and watched failing at -0.41 before the fix. `test_flat_group_does_not_poison_its_batch` pins that the mask is per row, and `test_flat_group_yields_no_update` pins the end-to-end consequence: a flat group produces exactly zero gradient. The committed `test_zero_variance_is_exact_zeros` passes under either check, which is why it did not catch this. `TestRewardDtype` pins the cast: bool and integer groups normalise to the same float advantages as their float equivalents, and a flat bool group is exact float zeros. It was added after a code review of the PR found that every earlier test passed float rewards only. `TestRewardGuards` pins the four loud failures (NaN, inf, an empty batch, a 0-dim input) and that a float16 group normalises finitely in float32; the same review found each of them.

## What grpo_loss reduces over

**Decision.**
`grpo_loss` returns the mean of `-advantage * logprob` over every element it is handed, as a 0-dim tensor for gradient descent, with the advantages detached inside. The gradient on each logprob is therefore `-advantage / n`, where n counts every element in the call.

**Rejected.**
The sum. The gradient magnitude then scales with the group size, so changing `--n-samples` changes the effective learning rate for no reason. The trainer clips the gradient norm, and the pre-clip norms it measured (the figure the module docstring records) sit far above the bound, so a sum would mostly be clipped away, but it would tie the pre-clip norm to n and hide the change behind the clip.
A per-group mean summed over groups. It differs from the plain mean by a constant equal to the number of groups, which is the same objection.
A per-token mean. This module receives one sequence logprob per rollout and takes it as given; whether that scalar is a token sum or a token mean is the harness's decision, nothing in this repo yet records which, and the module docstring's list of harness lessons names the choice and the 1/T it puts on the gradient scale.

**Evidence.**
(1) The gradient is linear in the advantage and independent of the logprob's value, so the reduction is the only thing that sets its scale.
(2) The direction is fixed by the committed `test_descent_direction`: descending the loss raises the logprob of a positive-advantage rollout.
(3) The advantages are detached because they are a function of the rewards, not of the policy, and the committed `test_advantages_detached` pins that no gradient reaches them.

**Cost accepted.**
The mean sees only what it is handed. A harness that accumulates backward per rollout, which is the lab's memory lesson, calls this function with one element at a time and so gets a sum across the group unless it scales each call by 1/G itself. That scaling lives at the trainer, alongside the clip, and this document is where the rule is recorded.

**Pinned by.**
`test_loss_is_the_mean_of_negative_weighted_logprobs` (-0.5, where a sum gives -1.0), `test_gradient_is_minus_advantage_over_n`, and the two scalar tests `test_loss_is_a_scalar` and `test_batched_rollouts_reduce_to_one_scalar`, all in `tests/test_grpo.py`. `TestLossShapeMismatch` pins that a (G, 1) column against a (G,) row raises rather than broadcasting to an outer product, and `TestLossGuards` that empty inputs raise rather than returning a NaN mean; the PR's code review found both.

## Forensics: obligation or convention

**Decision.**
Forensics is a harness responsibility, not an interface obligation: the protocol stays `score(candidates) -> list[float]` and nothing more. `evaluate` stores the per-problem record it already promises — samples, extracted candidates, scores, and the selected index — and that record answers the questions the lab actually had to ask. The one fact it cannot answer is per-stage attribution inside a composite, because summing stage scores discards the breakdown; that breakdown is accepted as lost, and if it is ever needed the composite can be re-run offline over the stored record, because every scorer is a pure function of its input ([What a scorer consumes and returns](#what-a-scorer-consumes-and-returns)) — provided that record holds the precomputed material, including the logprob summary, and not just the raw text. That requirement lands on `evaluate`'s record, not on the protocol.

**Rejected.**
Returning `(score, metadata)` from every scorer: it forces every call site, including the composite's inner loop, to unpack metadata it discards, and it makes the positional-alignment contract harder to state. Requiring an `explain()` method on the protocol: every implementation, including a three-line test fake, would have to write one for a question the lab has never yet needed to ask. A debug method on Composite alone: it would be a second way to call a scorer that only one class supports, and re-running the pure composite offline already recovers the same numbers.

**Evidence.**
(1) The recoverability test: with candidates and final scores stored, "why was this candidate not selected" is answerable — a`-inf` identifies a gate rejection — so the interface does not need tocarry that.
(2) The lab's selected-None-rate discovery needed only raw stored record, which is a harness capability, not a scorer one.
(3) What storage cannot recover is a stage-level breakdown: summing stage scores is a lossy combination, and the parts exist only inside `Composite`.
(4) `tests/test_scorers.py` committed three placeholder names covering the call contract, the veto, and the three roles — and none for forensics.

**Cost accepted.**
A composite's score is opaque about which stage produced it, so a future question of the form "did rank or vote agreement cost this candidate the selection" cannot be answered from stored records alone; answering it means re-running the composite over the stored candidates rather than reading it off the record.

**Pinned by.**
Nothing in `tests/test_scorers.py` — deliberately. This decision is pinned negatively: the scorer protocol has exactly one method, so `test_scorers_share_one_call_contract` passing with a three-line fake that implements only `score` is itself the evidence that forensics was not made an obligation.

## The contract tests

Four properties must hold for any implementation of this interface. The three test names already committed in `tests/test_scorers.py` stay as they are; these are what they assert.

**One call contract.**
Every scorer is callable with the candidate set alone, takes no ground-truth parameter, and returns a list the same length as its input.
This holds for a bare heuristic, for the logprob scorer, and for a `Composite`.
Enforced by `test_scorers_share_one_call_contract`, over a tuple of one plain scorer and one composite so that the composite is held to the same contract as its parts.

**The veto is absorbing.**
A candidate a gate rejected stays rejected no matter what follows it. Enforced by `test_parseability_gate_precedes_ranking`, composing the gate with an adversarial stage that returns `+inf` for everything: the rejected candidate still scores `-inf`. The test is written against a stage built to break the rule, not a well-behaved one, because the veto is a property of the combination rule rather than of stage behaviour.

**Scorers are pure.**
Calling `score` twice on the same record returns equal results, and the input list is unchanged afterwards. Enforced by a new test alongside the committed three. This one is load-bearing rather than defensive: the forensics decision rests entirely on being able to re-run a composite offline over the stored record, so purity is the assumption that has to be pinned, not assumed.

**An all-rejected set is data, not an error.**
When every candidate is rejected the composite returns a list of `-inf` and does not raise, so a composite can be nested inside another composite that treats that list as a legitimate intermediate result. Enforced by a case asserting the return value and, explicitly, the absence of an exception — thedeliberate counterpart to `self_consistency_vote`, which raises on empty input because it is a selector and must decide.

`test_same_verifier_serves_reward_and_validation_roles` covers the role split rather than a property of the scorer protocol: on a boxed sample all three roles return the same verdict, and on an unboxed sample the reward grader returns None while the eval grader rescues it —the roles differ exactly where extraction strictness applies and nowhere else.

House style follows the existing suite: class-grouped, one idea per test, concrete literal values rather than generated ones, and a comment naming the measurement that motivated the test.

## Stub contract audit

Every contract this repo committed before the interface existed, checked against the design. UNCHANGED rows name the design element that satisfies them; CHANGED rows record which side was wrong.

| # | Contract | Verdict |
|---|---|---|
| 1 | `extract_final_candidate(text, fallback=None)`, boxed-only default | UNCHANGED — *Three roles, one verifier* passes strictness per call and rests its fail-safe argument on this default, so the design depends on the promise rather than merely tolerating it. |
| 2 | A non-None fallback mode exists and rescues a bare number | UNCHANGED — the mode has a consumer, which is what the contract needs: the two eval roles use it. The mode is now named: `"number"` is the only lenient mode, and *The fallback contract* records what it rescues, what the official chain's second link was dropped for, and what an unknown mode does. |
| 3 | `verify_answer(candidate, ground_truth)`, None is always False | UNCHANGED — verification stays in `verifier.py` as plain functions, and this contract is the part all three roles share unchanged. |
| 4 | `self_consistency_vote(candidates)`: list in, winner out; None excluded pre-vote; empty raises; deterministic tiebreak | UNCHANGED — the contract holds untouched, but the design places a second vote-counting implementation beside it: `self_consistency_vote` returns a winner and owns the None-exclusion policy its module was assigned, while the composite's vote-agreement stage returns per-candidate agreement scores under the scorer signature. Recorded as accepted duplication, on the same grounds as the redundancy accepted in *Composition and the veto*: the two answer different questions, and neither may assume the other ran. |
| 5 | Group of one raises; zero-variance returns exact zeros | UNCHANGED — the reward path sits outside the scorer protocol, so no scorer ever hands a group to `grpo.py`. |
| 6 | `grpo_loss(logprobs, advantages)`: descent direction, advantages detached | UNCHANGED — `grpo.py` receives only logprobs and advantages; the scorer's logprob summary is built on the eval path instead, which is row 7. |
| 7 | `evaluate(model, tokenizer, problems, scorer)`, per-problem records | **CHANGED** — see Findings. |
| 8 | The three committed test names in `tests/test_scorers.py` | UNCHANGED — all three are writable under their existing names, and *The contract tests* adds two further tests without renaming any. |
| 9 | `--n-samples` default 7 | CHANGED (wording only) — the flag now also sets the candidate-set size handed to the composite scorer, not only to self-consistency. The help text "samples per problem for self-consistency" narrows what the flag controls, and is updated when the design lands. |

### Findings

**Row 7 — `evaluate` widens in two directions.**
The signature places the model asymmetry at the caller, so on the eval path `evaluate` must *build* the material bundle, including the logprob summary; and the offline re-run argument in *Forensics* requires that bundle to be *stored*, not just the raw text. The stub promises neither. It was under-specified rather than wrong: it was written before the material type existed. Both obligations land on `evaluate`, and neither reaches the protocol.

**Not in this table.**
The README's phrase "Scorer/reward is a swappable interface" overstates the design, since reward strictness swaps at the trainer harness rather than in `scorers.py`. That is prose rather than a committed contract, and it is corrected when the design lands.
