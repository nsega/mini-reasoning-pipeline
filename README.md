# mini-reasoning-pipeline

A minimal reasoning pipeline for Qwen3-0.6B where one verifier plays
three roles: **evaluation → reward → validation**.

> Status: private development. Goes public when the done-criteria hold:
> one-command end-to-end run, pipeline diagram, design rationale, and
> reproduction steps, all in this README.

## Pipeline

```
Qwen3-0.6B base
   → verifier (evaluation role)      · baseline eval on the frozen subset
   → self-consistency inference      · None-excluded voting
   → GRPO training (reward role)     · verifier as binary reward, no-KL default
   → verifier (validation role)      · before/after eval on the SAME subset
```

<!-- TODO: replace with the pipeline diagram (SVG) -->

## Design rationale

The design decisions here were earned in a 3-week study lab
(private repo) built around Sebastian Raschka's *Build a Reasoning
Model (From Scratch)*:

- **Selection is a swappable interface** (the hard constraint): no single
  signal survived ablation. Majority voting is poisoned by failed
  extractions pooling on one key; per-token logprob scoring over-selects
  degenerate fluency (a repetition loop is a confident place to be); a
  brevity heuristic skips long correct solutions. The interface composes
  signals — parseability gate first, then rank, then vote agreement.
  *Verification* is deliberately not part of it: a selection scorer never
  receives the ground truth, because two of its four call sites have none,
  and one that could see it could quietly cheat at eval time. Reward
  strictness therefore swaps at the trainer harness, not in `scorers.py`.
- **Scorers are pure functions of precomputed material**: the caller builds
  the bundle, so no scorer holds a model and the pure heuristics need no
  model double to test. Purity is also what makes a composite re-runnable
  offline over stored records, which is why the interface carries no
  reporting method of its own.
- **The veto is structural, not behavioural**: a gate rejects by scoring
  `-inf`, an absorbing element under the sum, so no later stage can lift a
  rejected candidate however it is written. Chosen over dropping, which
  would break the positional alignment the caller relies on.
- **One verifier, three roles, one difference**: evaluation, reward and
  validation share the same grader and extraction machinery; only
  extraction strictness differs. Reward is boxed-only, and boxed-only is
  the default, so a forgotten argument fails toward strictness rather than
  away from it.

The full record — every rejected option, its evidence, and the cost
accepted — is in [docs/design-notes.md](docs/design-notes.md), together
with an audit of the design against every contract this repo committed
before the interface existed.

## Reproduction

```bash
# TODO: the one command
uv run python run_pipeline.py
```

## Provenance

All pipeline code in `mini_reasoning/` is self-written (lab rule: only
self-written code migrates). Scaffolding, tests, and data scripts note
their origin in their headers. The evaluation subset derives from
MATH-500 (Hendrycks et al.; HuggingFaceH4/MATH-500), seed-fixed and
frozen.
