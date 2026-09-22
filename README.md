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
# The whole pipeline: baseline eval, self-consistency, GRPO, validation:
uv run python run_pipeline.py

# Or the inference and evaluation stages alone:
uv run python run_pipeline.py --skip-training
```

The evaluation subset is committed, so neither command regenerates it.
It is rebuilt only if the recipe itself ever changes, which is a thing
this experiment does not do mid-flight:

```bash
uv run --group data python data/make_subset.py
```

`datasets` sits in its own dependency group for that reason: it is a
heavy dependency that only the data script needs, so a clone that just
runs the pipeline never installs it. The group isolates installation,
not resolution: uv locks every group together, so datasets' own
constraints still reach `uv.lock` — they currently hold `fsspec`, a
transitive dependency of torch, one release behind where the pipeline
alone would put it. Nothing the pipeline needs is pinned by that, but
the group is not a second lockfile.

Expect roughly 70 seconds per problem for an eval at the default seven
samples and 512-token cap, so about an hour for the 50-problem subset
on CPU. A full run pays that twice, for the baseline and the
validation eval, and the training in between adds 40 steps of seven
rollouts each, plus a backward pass per rollout, since the trainer
runs them one at a time. Budget nearer three times a single eval, not
twice. Measured end to end on an M4 Pro at float32: 57 minutes for the
baseline eval, 44 for the 40 GRPO steps, 54 for the validation eval,
two and a half hours in all. The same work at the same seed has varied
by 25 minutes between runs on this machine, so these are a floor
rather than a clock. Per-problem cost tracks difficulty, because
harder problems run longer before they stop: samples average 666
characters at level 1 against 1187 at level 5, and this subset is
back-loaded with 36 of its 50 problems at levels 4 and 5. Training
holds the policy, its gradients and Adam's state at once, which for
this 0.6B model in float32 peaked at 8.6 GB.

A note on what the numbers mean: the validation eval grades with the
same verifier that produced the training reward, differing only in
extraction strictness. That circularity is deliberate and is the point
of the three-role design, but it means a before/after gain is a gain
against this grader, not an independent one.

## Provenance

All pipeline code in `mini_reasoning/` is self-written (lab rule: only
self-written code migrates). Scaffolding, tests, and data scripts note
their origin in their headers. The evaluation subset derives from
MATH-500 (Hendrycks et al.; HuggingFaceH4/MATH-500), seed-fixed and
frozen.
