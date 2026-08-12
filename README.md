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

- **Scorer/reward is a swappable interface** (the hard constraint):
  no single selection signal survived ablation. Majority voting is
  poisoned by failed extractions pooling on one key; per-token logprob
  scoring over-selects degenerate fluency (a repetition loop is a
  confident place to be); a brevity heuristic skips long correct
  solutions. The interface composes signals: parseability gate first,
  then rank, then vote agreement.
- **No-KL default for small-model sparse-reward GRPO**: with ~10%
  solve rates, most rollout groups are zero-advantage, and the simple
  signed-logratio KL penalty becomes the dominant gradient, uniformly
  suppressing the policy's own samples. Measured: the KL arm collapsed
  in 40 steps while the no-KL arm held steady.
- **Entropy and reference drift, not reward, are the training
  dashboard**: they moved first in every failure observed.
- **Level-reweighted reporting** for any comparison against published
  MATH-500 numbers: a 50-problem seed-fixed subset skews hard.

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
