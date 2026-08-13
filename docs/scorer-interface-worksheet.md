# Scorer interface — design worksheet

> Provenance: Claude-written scaffolding. It contains **questions, pointers to
> evidence already in this repo, and blanks** — no design decisions. Every
> answer is Naoki's. Delete this file once the answers land in
> `mini_reasoning/scorers.py` and the README's Design decisions section.

## How to use this

**Order.** Keep the reviewer's Q3 → Q1 → Q2 → Q4 → Q5. One amendment: spend the
first pass of Q3 on the framing question below (Q3.0), because its answer
silently decides Q1 and Q2.

**Method: design by call site, not by protocol.** You have four call sites and
all four are already sketched in the repo. Write the four calls first, as
pseudo-code, then read the signature off them. If a call site needs a comment
apologising for the shape, the signature is wrong.

```python
# 1. evaluation + validation roles — ground truth available, model available
#    mini_reasoning/evaluate.py:17
results = evaluate(model, tokenizer, problems, scorer)

# 2. reward role — ground truth available; per-token logprobs ALREADY computed
#    by the trainer, because grpo_loss needs them (mini_reasoning/grpo.py:25)
rewards = ...              # one scalar per rollout, group of n
advantages = group_relative_advantages(rewards)

# 3. selection at inference — NO ground truth exists at this point
#    mini_reasoning/consistency.py:17
winner = self_consistency_vote(candidates)

# 4. best-of-N under the composed scorer — NO ground truth at selection time;
#    ground truth is applied afterwards, only to grade the pick
winner = ...
```

**Stop at signatures.** Answer with `Protocol` / dataclass / docstring +
`raise NotImplementedError`. Do not implement while designing — it keeps the
closed-book rule intact and keeps the review cheap.

**Q5 is a checker, not a step.** After each of Q1–Q4, try to write the
assertion that would catch a violation. If you cannot write it, the decision is
still prose, not a contract. Collect the assertions as you go; Q5 is then just
selecting the best 2–3.

**Voice.** The destination is the README's Design rationale section, whose
existing bullets all have the shape *chose X over Y, because Z (measured)*.
Write every answer that way — name the rejected option.

**Done check.**
- All four call sites read naturally with no apology comment.
- The three existing placeholders in `tests/test_scorers.py` become real tests
  under their current names, with no renaming.
- Each answer names what it rejected.
- No already-specified stub contract had to change. If one did, that is a
  finding — record whether the design or the stub was wrong.

---

## Q3. Three roles, one object

### Q3.0 — the framing question (answer this first)

The stub calls this file "the scorer interface", and the README bullet
(`README.md:28`) calls it "Scorer/reward is a swappable interface". Those may be
two different things wearing one name:

- **Selection** ranks candidates against each other. Call sites 3 and 4.
  Ground truth **does not exist** at this point.
- **Verification** grades a candidate against ground truth. Call sites 1 and 2.
  Ground truth is **required** — `verify_answer(candidate, ground_truth)`,
  `verifier.py:26`.

The lab verdict quoted in the stub ("parseability gate → rank → vote agreement")
is entirely about **selection**. Q3's three roles (evaluation / reward /
validation) are entirely about **verification**. They meet at exactly one point:
the reward path, which selects and then grades.

Decide: is `scorers.py` **one protocol** (ground truth is an optional field of
the input bundle, `None` at selection time) or **two protocols** that compose?
State it in one sentence — the rest of the worksheet inherits it.

_Your answer:_

- 

### Q3.1 — what differs per role

Evidence in the repo:
- `verifier.py:5-13` — fallback is a **call-time argument**
  (`extract_final_candidate(text, fallback=None)`), and boxed-only is the
  **default**, so an omission bug fails toward strictness.
- `tests/test_verifier.py:23-25` pins that default; `tests/test_verifier.py:33`
  pins that some opt-in fallback exists.

Decide: does the scorer **bind** the extraction policy at construction (a
lenient eval instance and a strict reward instance are different objects) or
**pass** it per call (one instance, mode chosen by the caller)? Whichever you
pick, say how the fail-safe default survives it — the strict mode must still be
what you get when someone forgets to choose.

_Your answer (2–4 bullets):_

- 

---

## Q1. Signature and the placement of the asymmetry

Reviewer's three options: (a) scorer owns the model, (b) caller precomputes the
materials and every scorer is a pure function of them, (c) two protocols.

Evidence in the repo:
- `evaluate.py:17` — `evaluate(model, tokenizer, problems, scorer)` already
  passes model **and** scorer as separate parameters. Under (a) there would be
  two model references that can diverge. Is that acceptable or disqualifying?
- `grpo.py:25` — `grpo_loss(logprobs, advantages)`. The trainer **already**
  computes per-token logprobs, for the loss. In the reward path, the logprob
  scorer's "extra material" is therefore not extra at all — it is already in
  hand. Does that change the cost of (b)?
- `evaluate.py:13` — "Store per-problem records (samples, candidates, scores)".

**Output type — hard constraint.** GRPO consumes one scalar per rollout for a
whole group (`group_relative_advantages(rewards)`, `grpo.py:21`), not a winner.
Selection consumes a winner, not a vector. So the protocol must at minimum be
able to yield per-sample scores. Decide whether selection is then the caller's
argmax, or a second method on the protocol.

_Your answer (2–4 bullets):_

- 

---

## Q2. Composition, and what "veto" means

Decide: (i) does `Composite([...])` itself satisfy the protocol, and (ii) how is
the gate's veto made unbreakable by later stages?

**Stress case — work this one out explicitly, it is where the design bites.**
A group of n=7 rollouts, of which 6 fail parseability. Now compare the two
paths:

- Selection path: `consistency.py:6-10` says None candidates are **excluded**
  before voting, and that this belongs in the module, not the harness.
  `tests/test_consistency.py:20-24` pins it.
- Reward path: dropping the 6 leaves a group of 1, and
  `group_relative_advantages` raises ValueError on a group of one
  (`tests/test_grpo.py:29-31`). Keeping all 7 with the failures scored 0 gives a
  zero-variance group, which `tests/test_grpo.py:23-27` requires to be exactly
  zero — a safe no-op step.

So "gate rejects a candidate" appears to mean something different in the two
paths. Decide what your interface calls each, whether one mechanism can serve
both, and which layer owns the choice.

Second sub-question, inherited: if `self_consistency_vote` already drops None
internally **and** the composite's first stage is a parseability gate, the
filter runs twice. Is that defence in depth, or two owners for one policy?

_Your answer (2–4 bullets):_

- 

---

## Q4. Forensics — obligation or convention

Decide: is "enough bookkeeping to reanalyse offline" an **interface obligation**
(`(score, metadata)` return, or an `explain()` method) or a **harness
convention** (the scorer stays a plain function; `evaluate` stores everything)?

Evidence pulling toward convention:
- `evaluate.py:13` already assigns record-keeping to `evaluate`.
- `tests/test_scorers.py` has three placeholders, for Q1, Q2 and Q3 — **none for
  forensics**. Either the scaffold judged this a harness concern, or it is an
  oversight. Decide which, and say so.

Evidence pulling toward obligation:
- The lab's selected-None-rate discovery needed the *selection reason*, not just
  the raw candidates. Ask yourself concretely: with raw candidates alone, could
  you have recovered **which stage of a composite** rejected a given candidate?
  If not, storing inputs and outputs is not sufficient and the metadata has to
  come from inside.

_Your answer (2–4 bullets):_

- 

---

## Q5. Contract tests

The three names are already committed in `tests/test_scorers.py:11-23`:
`test_scorers_share_one_call_contract`,
`test_parseability_gate_precedes_ranking`,
`test_same_verifier_serves_reward_and_validation_roles`.
Treat them as a fixed spec — your design has to make all three writable under
those names.

House style, from the existing suite: class-grouped, one idea per test,
**concrete literal values** rather than property-based generation, and a comment
citing the measurement that motivated the test
(`tests/test_consistency.py:21-22` is the model to copy).

Pick 2–3 properties that must hold for *any* implementation, and express each as
a concrete case in that style. Candidates from the reviewer: determinism (same
input → same ranking), the gate's veto is unbreakable, a Composite's ranking
never contradicts its components. Add any that fell out of Q1–Q4 above.

_Your answer (2–3 properties, each with the assertion you would write):_

- 
